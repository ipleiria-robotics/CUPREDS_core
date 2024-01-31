/*
 MIT License

 Copyright (c) 2024 Carlos Cabaço Tojal

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 */

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

        // create the workers
        this->workersShouldStop = false;
        for(size_t i = 0; i < NUM_INTER_SENSOR_WORKERS; i++) {
            this->workers.emplace_back(&InterSensorManager::workersLoop, this);
        }
    }

    InterSensorManager::~InterSensorManager() {

        // free all the stream managers
        for(auto & streamManager : this->streamManagers) {
            streamManager.second.reset();
        }

        // wait for the memory monitoring thread
        this->keepThreadAlive = false;
        this->memoryMonitoringThread.join();

        // signal all workers to stop
        this->workersShouldStop = true;
        this->pendingCloudsCond.notify_all();
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

    void InterSensorManager::addSensorPointCloud(entities::StampedPointCloud cloud,
                                                 std::string& sensorName) {

        // create the entry
        // this is a shared pointer because it is shared among the queue and the map
        std::shared_ptr<struct pending_cloud_entry_t> newEntry = std::make_shared<struct pending_cloud_entry_t>(
                cloud,
                sensorName,
                0 // because will be added to the front
                );

        // acquire the mutex
        std::unique_lock lock(this->pendingCloudsMutex);

        // verify if the queue has space
        // if it doesn't, remove the oldest
        if(this->pendingCloudsQueue.size() == MAX_WORKER_QUEUE_LEN) {
            // get the last element
            std::shared_ptr<struct pending_cloud_entry_t>& toRemove = std::ref(this->pendingCloudsQueue.back());
            // remove from the map
            this->pendingCloudsBySensorName.erase(toRemove->sensorName);
            // remove from the queue
            this->pendingCloudsQueue.pop_back();
        }

        // if it was already in queue, remove the existent
        if(this->pendingCloudsBySensorName.contains(sensorName)) {

            // get a reference
            std::shared_ptr<struct pending_cloud_entry_t>& existent = std::ref(this->pendingCloudsBySensorName[sensorName]);
            size_t lookupIndex = existent->queueIndex;

            // remove the existent entry
            // from the queue
            this->pendingCloudsQueue.erase(this->pendingCloudsQueue.begin() + lookupIndex);
            // and from the map
            this->pendingCloudsBySensorName.erase(sensorName);

            // update subsequent entries
            for(size_t i = lookupIndex; i < this->pendingCloudsQueue.size(); i++) {
                (this->pendingCloudsQueue[i]->queueIndex)--;
            }
        }

        // add the entry to the front of the queue
        this->pendingCloudsQueue.push_front(newEntry);
        // add the entry to the map
        this->pendingCloudsBySensorName[sensorName] = newEntry;
        // release ownership from this method
        newEntry.reset();

        // notify the next worker that work is available
        this->pendingCloudsCond.notify_one();
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
        newStreamManager->setPointCloudReadyCallback(std::bind(&InterSensorManager::addSensorPointCloud, this,
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

    void InterSensorManager::workersLoop() {

        while(true) {

            entities::StampedPointCloud newCloud(POINTCLOUD_ORIGIN_NONE);

            {
                // acquire the queue mutex
                std::unique_lock lock(this->pendingCloudsMutex);

                // wait for an entry to be available / stop if signaled
                this->pendingCloudsCond.wait(lock, [this]() {
                    return !this->pendingCloudsQueue.empty() || this->workersShouldStop;
                });
                // if the workers should stop, do it
                if (this->workersShouldStop)
                    return;

                // pick an entry from the front of the queue
                std::shared_ptr<struct pending_cloud_entry_t>& work = std::ref(this->pendingCloudsQueue.front());
                std::string sensorName = work->sensorName;
                newCloud = work->cloud; // get the point cloud from the entry
                // remove from the queue
                this->pendingCloudsQueue.pop_front();
                // remove from the map
                this->pendingCloudsBySensorName.erase(sensorName);
            }

            {
                // lock the point cloud mutex
                std::unique_lock lock(this->cloudMutex);

                // register the point cloud
                this->mergedCloud.registerPointCloud(newCloud.getPointCloud());
            }

            // notify the next worker to start
            this->pendingCloudsCond.notify_one();
        }

    }
} // pcl_aggregator::managers
