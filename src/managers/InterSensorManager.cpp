//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/InterSensorManager.h>

namespace pcl_aggregator::managers {

    void memoryMonitoringRoutine(InterSensorManager *instance) {

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

    void streamCloudQuerierRoutine(InterSensorManager* instance, const std::string& topicName) {

        // query the last pointcloud of the IntraSensorManager by the topic name
        pcl::PointCloud<pcl::PointXYZRGBL> streamCloud = instance->streamManagers[topicName]->getCloud();

        // lock the merged cloud mutex
        std::unique_lock<std::mutex> lock(instance->cloudMutex);

        // wait for the cloud condition variable
        instance->cloudConditionVariable.wait(lock, [instance]{return instance->cloudReady;});

        // set cloud as not ready
        instance->cloudReady = false;

        // register the cloud
        instance->appendToMerged(streamCloud);

        // set the cloud as ready
        instance->cloudReady = true;

        // notify the next waiting thread
        instance->cloudConditionVariable.notify_one();

        //  sleep the thread considering the configured rate
        std::this_thread::sleep_for(std::chrono::duration<double>(1 / (double) instance->publishRate));
    }

    InterSensorManager::InterSensorManager(size_t nSources, double maxAge, size_t maxMemory, size_t publishRate):
    mergedCloud("mergedCloud") {
        this->nSources = nSources;

        this->maxAge = maxAge;
        this->maxMemory = maxMemory;
        this->publishRate = publishRate;

        // start the memory monitoring thread
        this->memoryMonitoringThread = std::thread(memoryMonitoringRoutine, this);
        pthread_setname_np(this->memoryMonitoringThread.native_handle(), "memory_monitoring_thread");
        this->memoryMonitoringThread.detach();
    }

    InterSensorManager::~InterSensorManager() {

        // free all the stream managers
        for(auto & streamManager : this->streamManagers) {
            streamManager.second.reset();
        }

        // wait for the memory monitoring thread
        this->keepThreadAlive = false;
        this->memoryMonitoringThread.join();
    }

    size_t InterSensorManager::getNClouds() const {
        return this->nSources;
    }

    void InterSensorManager::addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const std::string &topicName) {

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

    void InterSensorManager::setTransform(const Eigen::Affine3d &transform, const std::string &topicName) {
        this->initStreamManager(topicName, this->maxAge);

        this->streamManagers[topicName]->setSensorTransform(transform);
    }

    pcl::PointCloud<pcl::PointXYZRGBL> InterSensorManager::getMergedCloud() {
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

        // wait for the mutex
        std::unique_lock<std::mutex> lock(this->cloudMutex);

        // wait for the condition variable
        this->cloudConditionVariable.wait(lock, [this]{return this->cloudReady;});

        return *(this->mergedCloud.getPointCloud());
    }

    bool InterSensorManager::appendToMerged(pcl::PointCloud<pcl::PointXYZRGBL> input) {

        // TODO: move the registration implementation to the StampedPointCloud class

        bool couldAlign = false;

        // create a pointer from the pointcloud
        // linear complexity on the number of inserted points
        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr inputCloudPtr;
        inputCloudPtr->points.insert(inputCloudPtr->points.end(), input.points.begin(), input.points.end());

        // align the pointclouds
        if (!input.empty()) {

            {
                /* lock access to the pointcloud mutex by other threads.
                * will only be released after appending the input pointcloud. */
                std::unique_lock<std::mutex> lock(this->cloudMutex);

                // wait for the condition variable
                this->cloudConditionVariable.wait(lock, [this]{return this->cloudReady;});

                if (!this->mergedCloud.getPointCloud()->empty()) {

                    // create an ICP instance
                    pcl::IterativeClosestPoint<pcl::PointXYZRGBL, pcl::PointXYZRGBL> icp;
                    icp.setInputSource(this->mergedCloud.getPointCloud());
                    icp.setInputTarget(inputCloudPtr);

                    icp.setMaxCorrespondenceDistance(GLOBAL_ICP_MAX_CORRESPONDENCE_DISTANCE);
                    icp.setMaximumIterations(GLOBAL_ICP_MAX_ITERATIONS);

                    icp.align(
                            *this->mergedCloud.getPointCloud(), Eigen::Matrix4f::Identity()); // combine the aligned pointclouds on the "merged" instance

                    if (cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(),
                                                                      *inputCloudPtr) < 0) {
                        std::cerr << "Could not concatenate the pointclouds at the InterSensorManager!"
                                  << std::endl;
                    }

                    couldAlign = icp.hasConverged(); // return true if alignment was possible

                    /*
                    if(cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(), *input) < 0) {
                        std::cerr << "Could not concatenate the pointclouds at the InterSensorManager!" << std::endl;
                    }
                    couldAlign = false;
                     */

                } else {
                    if (cuda::pointclouds::concatenatePointCloudsCuda(this->mergedCloud.getPointCloud(), input) <
                        0) {
                        std::cerr << "Could not concatenate the pointclouds at the InterSensorManager!"
                                  << std::endl;
                    }
                }
            }

            // notify the waiting threads
            this->cloudConditionVariable.notify_one();
        }

        // the points are no longer needed
        input.clear();

        return couldAlign;
    }

    void InterSensorManager::removePointsByLabel(const std::set<std::uint32_t>& labels) {

        // remove the points with the label
        this->mergedCloud.removePointsWithLabels(labels);
    }

    void InterSensorManager::addStreamPointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud,
                                                 std::mutex& streamCloudMutex) {

        // DEPRECATED
        /*
        {
            std::lock_guard<std::mutex> lock(streamCloudMutex);
            this->appendToMerged(*cloud);
        }
        this->mergedCloud.downsample(VOXEL_LEAF_SIZE);
         */
    }

    void InterSensorManager::initStreamManager(const std::string &topicName, double maxAge) {
        std::lock_guard<std::mutex> lock(this->managersMutex);

        if(this->streamManagers.count(topicName) != 0)
            return;

        std::unique_ptr<IntraSensorManager> newStreamManager = std::make_unique<IntraSensorManager>(topicName, maxAge);

        // set the point removing method as a callback when some pointcloud ages on the stream manager
        newStreamManager->setPointAgingCallback(std::bind(&InterSensorManager::removePointsByLabel, this,
                                                          std::placeholders::_1));

        // add a pointcloud whenever the IntraSensorManager has one ready
        newStreamManager->setPointCloudReadyCallback(std::bind(&InterSensorManager::addStreamPointCloud, this,
                                                               std::placeholders::_1, std::placeholders::_2));

        this->streamManagers[topicName] = std::move(newStreamManager);

        // start the registration thread which runs at a configured rate
        std::thread streamRegistrationThread(streamCloudQuerierRoutine, this, topicName);

        // detach the registration thread
        streamRegistrationThread.detach();
    }

    void InterSensorManager::clearMergedCloud() {

        std::lock_guard<std::mutex> lock(this->cloudMutex);

        this->mergedCloud.getPointCloud()->clear();
    }

    InterSensorManager &InterSensorManager::get(size_t nSources, double maxAge, size_t maxMemory, size_t publishRate) {
        if(instance == nullptr)
            instance = new InterSensorManager(nSources, maxAge, maxMemory, publishRate);
        return *instance;
    }

    void InterSensorManager::destruct() {
        if(instance != nullptr) {
            delete instance;
            instance = nullptr;
        }
    }
} // pcl_aggregator::managers
