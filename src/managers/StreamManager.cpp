//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/StreamManager.h>
#include "pcl_aggregator_core/cuda/CUDAPointClouds.cuh"

namespace pcl_aggregator {
    namespace managers {

        void maxAgeWatchingRoutine(StreamManager* instance) {

            // lambda function which removes a pointcloud from the merged version
            auto pointCloudRemovalRoutine = [instance](std::set<std::uint32_t> labels) {
                instance->removePointClouds(labels);
            };

            std::set<std::uint32_t> labelsToRemove;

            while(instance->keepAgeWatcherAlive) {

                {
                    // lock access to the pointcloud set
                    std::lock_guard<std::mutex> lock(instance->setMutex);

                    for (auto &iter: instance->clouds) {

                        /* the set is ordered by ascending timestamp.
                         * When we find the first pointcloud which is not older than the max age, we can stop. */

                        // this pointcloud is older than the max age
                        if (iter->getTimestamp() <= utils::Utils::getMaxTimestampForAge(instance->maxAge)) {

                            // add the label to the set to remove
                            labelsToRemove.insert(iter->getLabel());
                        } else {
                            // the set is ordered by ascending timestamp, so we can stop here
                            break;
                        }

                        // TODO: review what happens to the pointer, potential memory leak here
                    }
                }


                // start a detached thread to the pointclouds
                /*
                 * using a deteched thread instead of sequentially to prevent from having this iteration
                 * going for too long, keeping access to the set constantly locked
                 */

                std::thread pointCloudRemovalThread = std::thread(pointCloudRemovalRoutine, labelsToRemove);
                pthread_setname_np(pointCloudRemovalThread.native_handle(), "pointCloudRemovalThread");
                pointCloudRemovalThread.detach();

                // the point aging callback was set
                if(instance->pointAgingCallback != nullptr) {
                    // call a thread to run the callback
                    // if it was done in the same thread, it would delay the routine
                    std::thread callbackThread = std::thread(instance->pointAgingCallback, labelsToRemove);
                    callbackThread.detach();
                }

                // sleep for a second before repeating
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        StreamManager::StreamManager(const std::string& topicName, double maxAge) {
            this->topicName = topicName;
            this->cloud = std::make_shared<entities::StampedPointCloud>(topicName);
            this->maxAge = maxAge;

            // start the age watcher thread
            this->maxAgeWatcherThread = std::thread(maxAgeWatchingRoutine, this);
            pthread_setname_np(this->maxAgeWatcherThread.native_handle(), "maxAgeWatcherThread");

            // this thread can dettach from the main thread
            this->maxAgeWatcherThread.detach();
        }

        StreamManager::~StreamManager() {
            this->cloud.reset();

            // signal the watcher to stop
            this->keepAgeWatcherAlive = false;
            // wait for the watcher to end
            this->maxAgeWatcherThread.join();

            for(const auto& c : this->clouds) {
                this->clouds.erase(c);
            }

            while(!this->cloudsNotTransformed.empty()) {
                this->cloudsNotTransformed.pop();
            }
        }

        bool StreamManager::operator==(const StreamManager &other) const {
            return this->topicName == other.topicName;
        }

        void StreamManager::computeTransform() {
            while(!this->cloudsNotTransformed.empty()) {

                // get the first element
                std::shared_ptr<entities::StampedPointCloud> spcl = this->cloudsNotTransformed.front();
                spcl->applyTransform(this->sensorTransform);

                // add to the set
                this->clouds.insert(std::move(spcl));

                // remove from the queue
                this->cloudsNotTransformed.pop();
            }
        }

        void StreamManager::removePointCloud(std::uint32_t label) {

            {
                std::lock_guard<std::mutex> cloudGuard(this->cloudMutex);

                // remove points with that label from the merged pointcloud
                this->cloud->removePointsWithLabel(label);
            }


            // lock the set
            std::lock_guard<std::mutex> guard(this->setMutex);

            // iterate the set
            for(auto c : this->clouds) {
                if(c->getLabel() == label) {
                    // remove the pointcloud from the set
                    this->clouds.erase(c);
                    c.reset();
                    break;
                }
            }

        }

        void StreamManager::removePointClouds(std::set<std::uint32_t> labels) {

            {
                std::lock_guard<std::mutex> cloudGuard(this->cloudMutex);

                // remove points with that label from the merged pointcloud
                this->cloud->removePointsWithLabels(labels);
            }


            // lock the set
            std::lock_guard<std::mutex> guard(this->setMutex);

            // iterate the set
            for(auto c : this->clouds) {
                if(labels.find(c->getLabel()) != labels.end()) {
                    // remove the pointcloud from the set
                    this->clouds.erase(c);
                    // free the pointcloud pointer
                    c.reset();
                }
            }

        }

        void StreamManager::addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr newCloud) {
            // check the incoming pointcloud for null or empty
            if(newCloud == nullptr)
                return;
            if(newCloud->empty()) {
                newCloud.reset();
                return;
            }

            // create a stamped point newCloud object to keep this pointcloud
            std::shared_ptr<entities::StampedPointCloud> spcl =
                    std::make_shared<entities::StampedPointCloud>(this->topicName);
            // the new pointcloud is moved to the StampedPointCloud
            spcl->setPointCloud(std::move(newCloud));

            if(!this->sensorTransformSet) {
                // add the pointcloud to the queue
                // the ownership is moved to the queue
                this->cloudsNotTransformed.push(std::move(spcl));
                return;
            }

            // transform the incoming pointcloud and add directly to the set

            // start a thread to transform the pointcloud
            auto transformRoutine = [this] (const std::shared_ptr<entities::StampedPointCloud>& spcl, const Eigen::Affine3d& tf) {
                applyTransformRoutine(this, spcl, tf);
            };

            // pointcloud is passed as a const reference: ownership is not moved and no copy is made
            std::thread transformationThread(transformRoutine, std::ref(spcl), sensorTransform);

            // start a thread to clear the pointclouds older than max age
            // std::thread cleaningThread(clearPointCloudsRoutine, this);

            // wait for both threads to synchronize
            transformationThread.join();
            // cleaningThread.join();

            // add the new pointcloud to the set
            /*
            int startingIndex = this->clouds.size();
            this->clouds.insert(spcl);
            int endingIndex = this->clouds.size();

            // store the indices in the merged cloud
            for(int i = startingIndex; i < endingIndex; i++) {
                spcl->addMergedIndex(i);
            }*/

            try {
                if(!spcl->getPointCloud()->empty()) {

                    {
                        std::lock_guard<std::mutex> cloudGuard(this->cloudMutex);

                        if (!this->cloud->getPointCloud()->empty()) {

                            /*
                            pcl::IterativeClosestPoint<pcl::PointXYZRGBL,pcl::PointXYZRGBL> icp;

                            icp.setInputSource(spcl->getPointCloud());
                            icp.setInputTarget(this->cloud->getPointCloud());

                            icp.setMaxCorrespondenceDistance(STREAM_ICP_MAX_CORRESPONDENCE_DISTANCE);
                            icp.setMaximumIterations(STREAM_ICP_MAX_ITERATIONS);

                            icp.align(*this->cloud->getPointCloud());

                            if (!icp.hasConverged()) {
                                *this->cloud->getPointCloud() += *spcl->getPointCloud(); // if alignment was not possible, just add the pointclouds
                            }

                            */

                            if (cuda::pointclouds::concatenatePointCloudsCuda(this->cloud->getPointCloud(),
                                                                              *(spcl->getPointCloud())) < 0) {
                                std::cerr << "Could not concatenate the pointclouds at the StreamManager!" << std::endl;
                            }

                        } else {
                            if (cuda::pointclouds::concatenatePointCloudsCuda(this->cloud->getPointCloud(),
                                                                              *(spcl->getPointCloud())) < 0) {
                                std::cerr << "Could not concatenate the pointclouds at the StreamManager!" << std::endl;
                            }
                        }

                        // downsample the new merged pointcloud
                        this->cloud->downsample(STREAM_DOWNSAMPLING_LEAF_SIZE);
                    }

                    // the points are no longer needed
                    spcl->getPointCloud()->clear();

                    if(this->pointCloudReadyCallback != nullptr) {

                        std::lock_guard<std::mutex> cloudGuard1(this->cloudMutex);

                        /*
                        // call the callback on a new thread
                         // WARNING: calling this thread as-is causes a race condition because the pointcloud is changed
                        std::thread pointCloudCallbackThread = std::thread(this->pointCloudReadyCallback,
                                                                           std::ref(*this->cloud->getPointCloud()));
                        pointCloudCallbackThread.detach();
                         */

                        this->pointCloudReadyCallback(std::ref(this->cloud->getPointCloud()));
                    }

                    /*
                    // start the pointcloud recycling thread
                    auto autoRemoveRoutine = [this] (
                                                 const std::shared_ptr<entities::StampedPointCloud>& spcl) {
                        pointCloudAutoRemoveRoutine(this, spcl);
                    };
                    std::thread spclRecyclingThread(autoRemoveRoutine, spcl);
                    // detach from the thread, this execution flow doesn't really care about it
                    spclRecyclingThread.detach(); */
                }

            } catch (std::exception &e) {
                std::cerr << "Error performing sensor-wise ICP: " << e.what() << std::endl;
            }

        }

        const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& StreamManager::getCloud() {
            std::lock_guard<std::mutex> lock(this->cloudMutex);
            return this->cloud->getPointCloud();
        }

        void StreamManager::setSensorTransform(const Eigen::Affine3d &transform) {

            std::lock_guard<std::mutex> lock(this->sensorTransformMutex);

            // set the new transform
            this->sensorTransform = transform;
            this->sensorTransformSet = true;
            this->computeTransform();
        }

        double StreamManager::getMaxAge() const {
            return this->maxAge;
        }

        void applyTransformRoutine(StreamManager *instance,
                                   const std::shared_ptr<entities::StampedPointCloud>& spcl,
                                   const Eigen::Affine3d& tf) {
            spcl->applyTransform(tf);
        }

        // NO LONGER USED: now, a single watching thread is used instead, to reduce overhead
        void pointCloudAutoRemoveRoutine(StreamManager* instance,
                                                std::shared_ptr<entities::StampedPointCloud> spcl) {

            spcl->getPointCloud().reset();

            // sleep for the max age
            std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<long long>(instance->maxAge * 1000)));

            // call the pointcloud removal method
            instance->removePointCloud(spcl->getLabel());
        }

        void icpTransformPointCloudRoutine(const std::shared_ptr<entities::StampedPointCloud>& spcl,
                                           const Eigen::Matrix4f& tf) {
            spcl->applyIcpTransform(tf);
        }

        std::function<void(std::set<std::uint32_t> labels)> StreamManager::getPointAgingCallback() const {
            return this->pointAgingCallback;
        }

        void StreamManager::setPointAgingCallback(const std::function<void(std::set<std::uint32_t>)>& func) {
            this->pointAgingCallback = func;
        }

        std::function<void(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud)>
        StreamManager::getPointCloudReadyCallback() const {
            return this->pointCloudReadyCallback;
        }

        void StreamManager::setPointCloudReadyCallback(
                const std::function<void(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &)> &func) {
            this->pointCloudReadyCallback = func;
        }


    } // pcl_aggregator
} // managers
