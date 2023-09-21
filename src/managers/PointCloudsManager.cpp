//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/PointCloudsManager.h>

namespace pcl_aggregator::managers {

    void memoryMonitoringRoutine(PointCloudsManager *instance) {

        while(instance->keepThreadAlive) {

            ssize_t pointsToRemove = 0;

            {
                std::lock_guard <std::mutex> lock(instance->cloudMutex);

                // get size in MB
                size_t cloudSize =
                        instance->mergedCloud.getPointCloud()->points.size() * sizeof(pcl::PointXYZRGBL) / (size_t) 1e6;

                // how many points need to be removed to match the maximum size or less?
                pointsToRemove = ceil(
                        (float) (cloudSize - instance->maxMemory) * 1e6 / sizeof(pcl::PointXYZRGBL));
            }

            if (pointsToRemove > 0) {

                std::cerr << "Exceeded the memory limit: " << pointsToRemove << " points will be removed!" << std::endl;

                auto pointRemoveRoutine = [instance](ssize_t pointsToRemove) {
                    std::lock_guard<std::mutex> lock(instance->cloudMutex);
                    // remove the points needed if the number of points exceed the maximum
                    for (size_t i = 0; i < pointsToRemove; i++)
                        instance->mergedCloud.getPointCloud()->points.pop_back();
                };

                // start a thread to remove the points and detach it
                std::thread removePointCount(pointRemoveRoutine, pointsToRemove);
                removePointCount.detach();
            }

            // this thread is repeating at a slow rate to prevent locking too much the mutex
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

    PointCloudsManager::PointCloudsManager(size_t nSources, double maxAge, size_t maxMemory):
    mergedCloud("mergedCloud") {
        this->nSources = nSources;

        this->maxAge = maxAge;
        this->maxMemory = maxMemory;

        // start the memory monitoring thread
        this->memoryMonitoringThread = std::thread(memoryMonitoringRoutine, this);
        pthread_setname_np(this->memoryMonitoringThread.native_handle(), "memory_monitoring_thread");
        this->memoryMonitoringThread.detach();
    }

    PointCloudsManager::~PointCloudsManager() {

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

    void PointCloudsManager::addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const std::string &topicName) {

        // check if the pointcloud is null or empty
        if(cloud == nullptr)
            return;
        if(cloud->empty()) {
            cloud.reset();
            return;
        }

        // the key is not present
        this->initStreamManager(topicName, this->maxAge);

        this->streamManagers[topicName]->addCloud(std::move(cloud));

    }

    void PointCloudsManager::setTransform(const Eigen::Affine3d &transform, const std::string &topicName) {
        this->initStreamManager(topicName, this->maxAge);

        this->streamManagers[topicName]->setSensorTransform(transform);
    }

    pcl::PointCloud<pcl::PointXYZRGBL> PointCloudsManager::getMergedCloud() {
        /*
        // clear the old merged cloud
        this->clearMergedCloud();

        bool firstCloud = true;
         */

        // iterate the map
        /*
        this->managersMutex.lock();
        for(auto & streamManager : this->streamManagers) {
            if(firstCloud) {
                this->cloudMutex.lock();
                *this->mergedCloud.getPointCloud() = *streamManager.second->getCloud();
                this->cloudMutex.unlock();
                firstCloud = false;
            } else {
                this->appendToMerged(streamManager.second->getCloud());
            }
        }
        this->managersMutex.unlock();*/

        std::lock_guard<std::mutex> lock(this->cloudMutex);
        return *(this->mergedCloud.getPointCloud());
    }

    bool PointCloudsManager::appendToMerged(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& input) {

        bool couldAlign = false;

        // align the pointclouds
        if (!input->empty()) {

            {
                /* lock access to the pointcloud mutex by other threads.
                * will only be released after appending the input pointcloud. */
                std::lock_guard<std::mutex> lock(this->cloudMutex);

                if (!this->mergedCloud.getPointCloud()->empty()) {

                    // create an ICP instance
                    pcl::IterativeClosestPoint<pcl::PointXYZRGBL, pcl::PointXYZRGBL> icp;
                    icp.setInputSource(this->mergedCloud.getPointCloud());
                    icp.setInputTarget(input);

                    icp.setMaxCorrespondenceDistance(GLOBAL_ICP_MAX_CORRESPONDENCE_DISTANCE);
                    icp.setMaximumIterations(GLOBAL_ICP_MAX_ITERATIONS);

                    icp.align(
                            *this->mergedCloud.getPointCloud(), Eigen::Matrix4f::Identity()); // combine the aligned pointclouds on the "merged" instance

                    if (cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(),
                                                                      *input) < 0) {
                        std::cerr << "Could not concatenate the pointclouds at the PointCloudsManager!"
                                  << std::endl;
                    }

                    couldAlign = icp.hasConverged(); // return true if alignment was possible

                    /*
                    if(cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(), *input) < 0) {
                        std::cerr << "Could not concatenate the pointclouds at the PointCloudsManager!" << std::endl;
                    }
                    couldAlign = false;
                     */

                } else {
                    if (cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(), *input) <
                        0) {
                        std::cerr << "Could not concatenate the pointclouds at the PointCloudsManager!"
                                  << std::endl;
                    }
                }
            }
        }

        // the points are no longer needed
        input->clear();

        return couldAlign;
    }

    void PointCloudsManager::removePointsByLabel(const std::set<std::uint32_t>& labels) {

        // remove the points with the label
        this->mergedCloud.removePointsWithLabels(labels);
    }

    void PointCloudsManager::addStreamPointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud,
                                                 std::mutex& streamCloudMutex) {

        {
            std::lock_guard<std::mutex> lock(streamCloudMutex);
            this->appendToMerged(cloud);
        }
        this->mergedCloud.downsample(VOXEL_LEAF_SIZE);
    }

    void PointCloudsManager::initStreamManager(const std::string &topicName, double maxAge) {
        std::lock_guard<std::mutex> lock(this->managersMutex);

        if(this->streamManagers.count(topicName) != 0)
            return;

        std::unique_ptr<StreamManager> newStreamManager = std::make_unique<StreamManager>(topicName, maxAge);

        // set the point removing method as a callback when some pointcloud ages on the stream manager
        newStreamManager->setPointAgingCallback(std::bind(&PointCloudsManager::removePointsByLabel, this,
                                                          std::placeholders::_1));

        // add a pointcloud whenever the StreamManager has one ready
        newStreamManager->setPointCloudReadyCallback(std::bind(&PointCloudsManager::addStreamPointCloud, this,
                                                               std::placeholders::_1, std::placeholders::_2));

        this->streamManagers[topicName] = std::move(newStreamManager);
    }

    void PointCloudsManager::clearMergedCloud() {

        std::lock_guard<std::mutex> lock(this->cloudMutex);

        this->mergedCloud.getPointCloud()->clear();
    }
} // pcl_aggregator::managers
