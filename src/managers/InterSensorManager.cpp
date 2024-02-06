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

    InterSensorManager* InterSensorManager::instance = nullptr;

    InterSensorManager::InterSensorManager(size_t nSources, double maxAge):
    mergedCloud("mergedCloud") {
        this->nSources = nSources;

        this->maxAge = maxAge;

        // create the workers
        this->workersShouldStop = false;
        for(size_t i = 0; i < NUM_INTER_SENSOR_WORKERS; i++) {
            this->workers.emplace_back(&InterSensorManager::workersLoop, this);
        }
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

        std::cout << "Adding cloud" << std::endl;

        // the key is not present
        this->initStreamManager(topicName, this->maxAge);

        this->streamManagers[topicName]->addCloud(std::move(cloud));

    }

    void InterSensorManager::setTransform(const Eigen::Affine3d &transform, const std::string &topicName) {
        this->initStreamManager(topicName, this->maxAge);

        this->streamManagers[topicName]->setSensorTransform(transform);
    }

    pcl::PointCloud<pcl::PointXYZRGBL> InterSensorManager::getMergedCloud() {

        pcl::PointCloud<pcl::PointXYZRGBL> tmp;

        {
            // wait for the mutex
            std::unique_lock<std::mutex> lock(this->cloudMutex);

            // wait for the condition variable
            this->cloudConditionVariable.wait(lock, [this] { return this->cloudReady; });

            tmp = *(this->mergedCloud.getPointCloud());
        }

        // notify the next thread in queue to continue
        this->cloudConditionVariable.notify_one();

        return tmp;
    }

    void InterSensorManager::removePointsByLabel(const std::set<std::uint32_t>& labels) {

        std::thread pointRemovingThread = std::thread([this,labels]() {

            {
                // lock the point cloud
                std::unique_lock lock(this->cloudMutex);

                // wait for the condition variable
                this->cloudConditionVariable.wait(lock, [this]() {
                    return this->cloudReady;
                });

                this->cloudReady = false;

                // remove the points with the label
                this->mergedCloud.removePointsWithLabels(labels);

                this->cloudReady = true;
            }

            this->cloudConditionVariable.notify_one();
        });
        pointRemovingThread.detach();
    }

    void InterSensorManager::addSensorPointCloud(entities::StampedPointCloud cloud,
                                                 std::string& sensorName) {

        // create the entry
        // this is a shared pointer because it is shared among the queue and the map
        struct pending_cloud_entry_t e = {
                cloud,
                sensorName,
                0 // because will be added to the front
        };
        std::shared_ptr<struct pending_cloud_entry_t> newEntry = std::make_shared<struct pending_cloud_entry_t>(e);

        {
            // acquire the mutex
            std::unique_lock lock(this->pendingCloudsMutex);

            this->pendingCloudsCond.wait(lock);

            // verify if the queue has space
            // if it doesn't, remove the oldest
            if (this->pendingCloudsQueue.size() == MAX_WORKER_QUEUE_LEN) {
                // get the last element
                std::shared_ptr<struct pending_cloud_entry_t> &toRemove = std::ref(this->pendingCloudsQueue.back());
                // remove from the map
                this->pendingCloudsBySensorName.erase(toRemove->sensorName);
                // remove from the queue
                this->pendingCloudsQueue.pop_back();
            }

            // if it was already in queue, remove the existent
            if (this->pendingCloudsBySensorName.contains(sensorName)) {

                // get a reference
                std::shared_ptr<struct pending_cloud_entry_t> &existent = std::ref(
                        this->pendingCloudsBySensorName[sensorName]);
                size_t lookupIndex = existent->queueIndex;

                // remove the existent entry
                // from the queue
                this->pendingCloudsQueue.erase(this->pendingCloudsQueue.begin() + lookupIndex);
                // and from the map
                this->pendingCloudsBySensorName.erase(sensorName);

                // update subsequent entries
                for (size_t i = lookupIndex; i < this->pendingCloudsQueue.size(); i++) {
                    (this->pendingCloudsQueue[i]->queueIndex)--;
                }
            }

            // add the entry to the front of the queue
            this->pendingCloudsQueue.push_front(newEntry);
            // add the entry to the map
            this->pendingCloudsBySensorName[sensorName] = newEntry;
            // release ownership from this method
            newEntry.reset();
        }

        // notify the next worker that work is available
        this->pendingCloudsCond.notify_one();
    }

    void InterSensorManager::initStreamManager(const std::string &topicName, double maxAge) {
        std::lock_guard<std::mutex> lock(this->managersMutex);

        std::cout << "Will the manager initialize?" << std::endl;

        if(this->streamManagers.count(topicName) != 0)
            return;

        std::cout << "Init new Intra-sensor manager" << std::endl;

        std::unique_ptr<IntraSensorManager> newStreamManager = std::make_unique<IntraSensorManager>(topicName, maxAge);

        // set the point removing method as a callback when some pointcloud ages on the stream manager
        newStreamManager->setPointAgingCallback(std::bind(&InterSensorManager::removePointsByLabel, this,
                                                          std::placeholders::_1));

        // add a pointcloud whenever the IntraSensorManager has one ready
        newStreamManager->setPointCloudReadyCallback(std::bind(&InterSensorManager::addSensorPointCloud, this,
                                                               std::placeholders::_1, std::placeholders::_2));

        this->streamManagers[topicName] = std::move(newStreamManager);
    }

    void InterSensorManager::clearMergedCloud() {

        std::lock_guard<std::mutex> lock(this->cloudMutex);

        this->mergedCloud.getPointCloud()->clear();
    }

    InterSensorManager& InterSensorManager::getInstance(size_t nSources, double maxAge) {
        if(instance == nullptr)
            instance = new InterSensorManager(nSources, maxAge);
        return *instance;
    }

    void InterSensorManager::destruct() {
        if(instance != nullptr) {
            // free all the stream managers
            for(auto & streamManager : instance->streamManagers) {
                streamManager.second.reset();
            }

            // signal all workers to stop
            instance->workersShouldStop = true;
            instance->pendingCloudsCond.notify_all();

            delete instance;
            instance = nullptr;
        }
    }

    double InterSensorManager::getAverageRegistrationTime() {
        double val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            // wait for the condition variable
            this->statisticsCond.wait_for(lock, std::chrono::milliseconds(50));

            val = this->avgRegistrationTimeMs;
        }

        // notify the next thread
        this->statisticsCond.notify_one();

        return val;
    }

    double InterSensorManager::getVarianceRegistrationTime() {
        double val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            // wait for the condition variable
            this->statisticsCond.wait_for(lock, std::chrono::milliseconds(50));

            val = this->varRegistrationTimeMs;
        }

        // notify the next thread
        this->statisticsCond.notify_one();

        return val;
    }

    size_t InterSensorManager::getSampleCount() {
        size_t val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            // wait for the condition variable
            this->statisticsCond.wait_for(lock, std::chrono::milliseconds(50));

            val = this->registrationTimeSampleCount;
        }

        // notify the next thread
        this->pendingCloudsCond.notify_one();

        return val;
    }

    void InterSensorManager::workersLoop() {

        while(true) {

            std::cout << "Inter-sensor worker look start" << std::endl;

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

            // notify next waiting thread
            this->pendingCloudsCond.notify_one();

            {
                // lock the point cloud mutex
                std::unique_lock lock(this->cloudMutex);

                // wait for the point cloud condition variable
                this->cloudConditionVariable.wait(lock, [this]() {
                    return this->cloudReady;
                });

                this->cloudReady = false;

                // register the point cloud
                this->mergedCloud.registerPointCloud(newCloud.getPointCloud());

                this->cloudReady = true;
            }

            // notify next waiting thread
            this->cloudConditionVariable.notify_one();

            // update the statistics
            auto now = utils::Utils::getCurrentTimeMillis();
            unsigned long long diff = now - newCloud.getTimestamp();
            {
                // acquire the mutex
                std::unique_lock lock(this->statisticsMutex);

                // wait for the statistics condition variable
                this->statisticsCond.wait_for(lock, std::chrono::milliseconds(50));

                // update the statistics
                double delta = (double) diff - this->avgRegistrationTimeMs;
                (this->registrationTimeSampleCount)++;

                this->avgRegistrationTimeMs =
                        this->avgRegistrationTimeMs + delta / (double) this->registrationTimeSampleCount;
                if(this->registrationTimeSampleCount >= 2)
                    this->varRegistrationTimeMs =
                            this->varRegistrationTimeMs + delta * ((double) diff - this->avgRegistrationTimeMs);
            }

            // notify the next thread waiting for statistics
            this->statisticsCond.notify_one();

            std::cout << "Inter-sensor worker loop end" << std::endl;
        }

    }
} // pcl_aggregator::managers
