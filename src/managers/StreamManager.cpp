//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/StreamManager.h>

namespace pcl_aggregator {
    namespace managers {

        template<typename LabeledPointTypeT>
        StreamManager<LabeledPointTypeT>::StreamManager(std::string topicName, double maxAge) {
            this->topicName = topicName;
            this->cloud = std::make_shared<entities::StampedPointCloud<LabeledPointTypeT>>(topicName);
            this->maxAge = maxAge;
        }

        template<typename LabeledPointTypeT>
        StreamManager<LabeledPointTypeT>::~StreamManager() {
            this->cloud.reset();
        }

        template<typename LabeledPointTypeT>
        bool StreamManager<LabeledPointTypeT>::operator==(const StreamManager &other) const {
            return this->topicName == other.topicName;
        }

        template<typename LabeledPointTypeT>
        void StreamManager<LabeledPointTypeT>::computeTransform() {
            while(this->cloudsNotTransformed.size() > 0) {

                // get the first element
                std::shared_ptr<entities::StampedPointCloud<LabeledPointTypeT>> spcl = this->clouds_not_transformed.front();
                spcl->applyTransform(this->sensorTransform);

                // add to the set
                this->clouds.insert(spcl);

                // remove from the queue
                this->clouds_not_transformed.pop();
            }

            this->pointCloudSet = true;
        }

        template<typename LabeledPointTypeT>
        void StreamManager<LabeledPointTypeT>::removePointCloud(std::shared_ptr<entities::StampedPointCloud<LabeledPointTypeT>> spcl) {
            // remove points with that label from the merged pointcloud
            this->cloud->removePointsWithLabel(spcl->getLabel());

            // lock the set
            std::lock_guard<std::mutex> guard(this->setMutex);

            // iterate the set
            for(auto it = this->clouds.begin(); it != this->clouds.end(); it++) {
                if((*it)->getLabel() == spcl->getLabel()) {
                    // remove the pointcloud from the set
                    this->clouds.erase(it);
                    it->reset();
                }
            }
        }

        template<typename LabeledPointTypeT>
        void StreamManager<LabeledPointTypeT>::addCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud) {
            // check the incoming pointcloud for null or empty
            if(cloud == nullptr)
                return;
            if(cloud->empty())
                return;

            // create a stamped point cloud object to keep this pointcloud
            std::shared_ptr<entities::StampedPointCloud<LabeledPointTypeT>> spcl =
                    std::make_shared<entities::StampedPointCloud<LabeledPointTypeT>>(this->topicName);
            spcl->setPointCloud(cloud);

            if(!this->sensorTransformSet) {
                // add the pointcloud to the queue
                this->clouds_not_transformed.push(spcl);
                return;
            }

            // transform the incoming pointcloud and add directly to the set

            // start a thread to transform the pointcloud
            auto transformRoutine = [] (StreamManager* instance, const std::shared_ptr<entities::StampedPointCloud<LabeledPointTypeT>>& spcl, const Eigen::Affine3d& tf) {
                applyTransformRoutine(instance, spcl, tf);
            };
            std::thread transformationThread(transformRoutine, this, spcl, sensorTransform);

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
                        pcl::IterativeClosestPoint<LabeledPointTypeT,LabeledPointTypeT> icp;

                        icp.setInputSource(spcl->getPointCloud());
                        icp.setInputTarget(this->cloud->getPointCloud());

                        icp.setMaxCorrespondenceDistance(STREAM_ICP_MAX_CORRESPONDENCE_DISTANCE);
                        icp.setMaximumIterations(STREAM_ICP_MAX_ITERATIONS);

                        icp.align(*this->cloud->getPointCloud());

                        if (!icp.hasConverged())
                            *this->cloud->getPointCloud() += *spcl->getPointCloud(); // if alignment was not possible, just add the pointclouds

                    } else {
                        *this->cloud->getPointCloud() = *spcl->getPointCloud();
                    }
                    this->cloudMutex.unlock();

                    // remove the points. they are not needed, just the label
                    spcl->getPointCloud().reset();

                    // start the pointcloud recycling thread
                    auto autoRemoveRoutine = [] (StreamManager* instance,
                                                 const std::shared_ptr<entities::StampedPointCloud<LabeledPointTypeT>>& spcl) {
                        pointCloudAutoRemoveRoutine(instance, spcl);
                    };
                    std::thread spclRecyclingThread(autoRemoveRoutine, this, spcl);
                    // detach from the thread, this execution flow doesn't really care about it
                    spclRecyclingThread.detach();
                }

            } catch (std::exception &e) {
                std::cout << "Error performing sensor-wise ICP: " << e.what() << std::endl;
            }

        }

        template<typename LabeledPointTypeT>
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr StreamManager<LabeledPointTypeT>::getCloud() const {
            return this->cloud;
        }

        template<typename LabeledPointTypeT>
        void StreamManager<LabeledPointTypeT>::setSensorTransform(const Eigen::Affine3d &transform) {
            this->sensorTransform = transform;
        }

        template<typename LabeledPointTypeT>
        double StreamManager<LabeledPointTypeT>::getMaxAge() const {
            return this->maxAge;
        }

        template <typename RoutinePointTypeT>
        void applyTransformRoutine(StreamManager<RoutinePointTypeT> *instance,
                                   const std::shared_ptr<entities::StampedPointCloud<RoutinePointTypeT>>& spcl,
                                   const Eigen::Affine3d& tf) {
            spcl->applyTransform(tf);
        }

        template <typename RoutinePointTypeT>
        void pointCloudAutoRemoveRoutine(StreamManager<RoutinePointTypeT>* instance,
                                                const std::shared_ptr<entities::StampedPointCloud<RoutinePointTypeT>>& spcl) {
            // sleep for the max age
            std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<long long>(instance->max_age * 1000)));

            // call the pointcloud removal method
            instance->removePointCloud(spcl);

            // free the pointer
            spcl.reset();
        }

        template <typename RoutinePointTypeT>
        void icpTransformPointCloudRoutine(const std::shared_ptr<entities::StampedPointCloud<RoutinePointTypeT>>& spcl,
                                           const Eigen::Matrix4f& tf) {
            spcl->applyIcpTransform(tf);
        }


    } // pcl_aggregator
} // managers