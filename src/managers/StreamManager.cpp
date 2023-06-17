//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/StreamManager.h>

namespace pcl_aggregator {
    namespace managers {

        void maxAgeWatchingRoutine(StreamManager* instance) {

            // lambda function which removes a pointcloud from the merged version
            auto pointCloudRemovalRoutine = [instance](std::uint32_t label) {
                instance->cloud->removePointsWithLabel(label);
            };

            while(instance->keepAgeWatcherAlive) {

                // lock access to the pointcloud set
                instance->setMutex.lock();

                for(auto& iter : instance->clouds) {

                    // this pointcloud is older than the max age
                    if(utils::Utils::getAgeInSecs(iter->getTimestamp()) >= instance->maxAge) {
                        // start a detached thread to remove it
                        /*
                         * using a deteched thread instead of sequentially to prevent from having this iteration
                         * going for too long, keeping access to the set constantly locked
                         */
                        std::thread pointCloudRemovalThread = std::thread(pointCloudRemovalRoutine,
                                                                          iter->getLabel());
                        pointCloudRemovalThread.detach();

                        // the point aging callback was set
                        if(instance->pointAgingCallback != nullptr) {
                            // call a thread to run the callback
                            std::thread callbackThread = std::thread(instance->pointAgingCallback, iter->getLabel());
                            callbackThread.detach();
                        }
                    }
                }

                // unlock access to hte set
                instance->setMutex.unlock();

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

            // this thread can dettach from the main thread
            this->maxAgeWatcherThread.detach();
        }

        StreamManager::~StreamManager() {
            this->cloud.reset();

            // signal the watcher to stop
            this->keepAgeWatcherAlive = false;
            // wait for the watcher to end
            this->maxAgeWatcherThread.join();
        }

        bool StreamManager::operator==(const StreamManager &other) const {
            return this->topicName == other.topicName;
        }

        void StreamManager::computeTransform() {
            while(this->cloudsNotTransformed.size() > 0) {

                // get the first element
                std::shared_ptr<entities::StampedPointCloud> spcl = this->cloudsNotTransformed.front();
                spcl->applyTransform(this->sensorTransform);

                // add to the set
                this->clouds.insert(spcl);

                // remove from the queue
                this->cloudsNotTransformed.pop();
            }
        }

        void StreamManager::removePointCloud(std::shared_ptr<entities::StampedPointCloud> spcl) {

            this->cloudMutex.lock();
            // remove points with that label from the merged pointcloud
            this->cloud->removePointsWithLabel(spcl->getLabel());
            this->cloudMutex.unlock();


            // lock the set
            std::lock_guard<std::mutex> guard(this->setMutex);

            // iterate the set
            for(const auto& c : this->clouds) {
                if(c->getLabel() == spcl->getLabel()) {
                    // remove the pointcloud from the set
                    this->clouds.erase(c);
                }
            }

            spcl.reset();

        }

        void StreamManager::addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud) {
            // check the incoming pointcloud for null or empty
            if(cloud == nullptr)
                return;
            if(cloud->empty())
                return;

            // create a stamped point cloud object to keep this pointcloud
            std::shared_ptr<entities::StampedPointCloud> spcl =
                    std::make_shared<entities::StampedPointCloud>(this->topicName);
            spcl->setPointCloud(std::move(cloud));

            if(!this->sensorTransformSet) {
                // add the pointcloud to the queue
                this->cloudsNotTransformed.push(std::move(spcl));
                return;
            }

            // transform the incoming pointcloud and add directly to the set

            // start a thread to transform the pointcloud
            auto transformRoutine = [this] (const std::shared_ptr<entities::StampedPointCloud>& spcl, const Eigen::Affine3d& tf) {
                applyTransformRoutine(this, spcl, tf);
            };
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
                    this->cloudMutex.lock();
                    if(!this->cloud->getPointCloud()->empty()) {

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

                        *this->cloud->getPointCloud() += *spcl->getPointCloud();

                    } else {
                        *this->cloud->getPointCloud() = *spcl->getPointCloud();
                    }

                    if(this->pointCloudReadyCallback != nullptr) {

                        // call the callback on a new thread
                        std::thread pointCloudCallbackThread = std::thread(this->pointCloudReadyCallback,
                                                                           *this->cloud->getPointCloud());
                        pointCloudCallbackThread.detach();
                    }

                    this->cloudMutex.unlock();

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
                std::cout << "Error performing sensor-wise ICP: " << e.what() << std::endl;
            }

        }

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr StreamManager::getCloud() const {
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

        void pointCloudAutoRemoveRoutine(StreamManager* instance,
                                                const std::shared_ptr<entities::StampedPointCloud>& spcl) {

            spcl->getPointCloud().reset();

            // sleep for the max age
            std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<long long>(instance->maxAge * 1000)));

            // call the pointcloud removal method
            instance->removePointCloud(spcl);
        }

        void icpTransformPointCloudRoutine(const std::shared_ptr<entities::StampedPointCloud>& spcl,
                                           const Eigen::Matrix4f& tf) {
            spcl->applyIcpTransform(tf);
        }

        std::function<void(std::uint32_t label)> StreamManager::getPointAgingCallback() const {
            return this->pointAgingCallback;
        }

        void StreamManager::setPointAgingCallback(const std::function<void(std::uint32_t)>& func) {
            this->pointAgingCallback = func;
        }

        std::function<void(pcl::PointCloud<pcl::PointXYZRGBL> &cloud)>
        StreamManager::getPointCloudReadyCallback() const {
            return this->pointCloudReadyCallback;
        }

        void StreamManager::setPointCloudReadyCallback(
                const std::function<void(const pcl::PointCloud<pcl::PointXYZRGBL> &)> &func) {
            this->pointCloudReadyCallback = func;
        }


    } // pcl_aggregator
} // managers