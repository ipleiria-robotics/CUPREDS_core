//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/PointCloudsManager.h>

namespace pcl_aggregator {
    namespace managers {

        void memoryMonitoringRoutine(PointCloudsManager *instance) {

            while(instance->keepThreadAlive) {
                // lock access to the pointcloud
                instance->cloudMutex.lock();

                // get size in MB
                size_t cloudSize = instance->mergedCloud->points.size() * sizeof(pcl::PointXYZRGBL) / 1000000;

                // how many points need to be removed to match the maximum size or less?
                ssize_t pointsToRemove = ceil(
                        (float) (cloudSize - instance->maxMemory) * 1000000 / sizeof(pcl::PointXYZRGBL));

                if (pointsToRemove > 0) {
                    // remove the points needed if the number of points exceed the maximum
                    for (size_t i = 0; i < pointsToRemove; i++)
                        instance->mergedCloud->points.pop_back();
                }

                instance->cloudMutex.unlock();

                // run the thread routine each second
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        PointCloudsManager::PointCloudsManager(size_t nSources, double maxAge, size_t maxMemory) {
            this->nSources = nSources;

            // initialize empty merged cloud
            this->mergedCloud = pcl::PointCloud<pcl::PointXYZRGBL>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBL>());

            this->maxAge = maxAge;
            this->maxMemory = maxMemory;

            // start the memory monitoring thread
            this->memoryMonitoringThread = std::thread(memoryMonitoringRoutine, this);
            this->memoryMonitoringThread.detach();
        }

        PointCloudsManager::~PointCloudsManager() {
            // free the merged pointcloud
            this->mergedCloud.reset();

            // free all the stream managers
            for(auto & streamManager : this->streamManagers) {
                streamManager.second.reset();
            }

            // wait for the memory monitoring thread
            this->keepThreadAlive = false;
            this->memoryMonitoringThread.join();
        }

        size_t PointCloudsManager::getNClouds() const {
            return this->nSources;
        }

        void PointCloudsManager::addCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud, const std::string &topicName) {

            // check if the pointcloud is null or empty
            if(cloud == nullptr)
                return;
            if(cloud->empty())
                return;

            // the key is not present
            this->initStreamManager(topicName, this->maxAge);

            this->streamManagers[topicName]->addCloud(cloud);

        }

        void PointCloudsManager::setTransform(const Eigen::Affine3d &transform, const std::string &topicName) {
            this->initStreamManager(topicName, this->maxAge);

            this->streamManagers[topicName]->setSensorTransform(transform);
        }

        pcl::PointCloud<pcl::PointXYZRGBL> PointCloudsManager::getMergedCloud() {
            // clear the old merged cloud
            this->clearMergedCloud();

            bool firstCloud = true;

            // iterate the map
            /* TODO: review performance of only perform merging on demand
             * vs merging the pointclouds and removing as needed every time
            */
            this->managersMutex.lock();
            for(auto & streamManager : this->streamManagers) {
                if(firstCloud) {
                    this->cloudMutex.lock();
                    this->mergedCloud = streamManager.second->getCloud();
                    this->cloudMutex.unlock();
                    firstCloud = false;
                } else {
                    this->appendToMerged(streamManager.second->getCloud());
                }
            }
            this->managersMutex.unlock();

            // this->downsampleMergedCloud();
            return *this->mergedCloud;
        }

        bool PointCloudsManager::appendToMerged(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &input) {

            this->cloudMutex.lock();

            // align the pointclouds
            if(!input->empty()) {
                if(!this->mergedCloud->empty()) {
                    /*
                    // create an ICP instance
                    pcl::IterativeClosestPoint<pcl::PointXYZRGBL, pcl::PointXYZRGBL> icp;
                    icp.setInputSource(input);
                    icp.setInputTarget(this->mergedCloud); // "input" will align to "merged"

                    icp.setMaxCorrespondenceDistance(GLOBAL_ICP_MAX_CORRESPONDENCE_DISTANCE);
                    icp.setMaximumIterations(GLOBAL_ICP_MAX_ITERATIONS);

                    icp.align(*this->mergedCloud); // combine the aligned pointclouds on the "merged" instance

                    if (!icp.hasConverged())
                        *this->mergedCloud += *input; // if alignment was not possible, just add the pointclouds

                    return icp.hasConverged(); // return true if alignment was possible */

                    *this->mergedCloud += *input;
                    return false;

                } else {
                    *this->mergedCloud = *input;
                }

            }

            this->cloudMutex.unlock();

            return false;
        }

        void PointCloudsManager::initStreamManager(const std::string &topicName, double maxAge) {
            std::lock_guard<std::mutex> lock(this->managersMutex);

            if(this->streamManagers.count(topicName) != 0)
                return;
            this->streamManagers[topicName] = std::make_unique<StreamManager>(topicName, maxAge);
        }

        void PointCloudsManager::clearMergedCloud() {

            this->cloudMutex.lock();
            this->mergedCloud->clear();
            this->cloudMutex.unlock();
        }

    } // pcl_aggregator
} // managers
