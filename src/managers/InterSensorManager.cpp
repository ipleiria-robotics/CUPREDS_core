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

        // create the removal worker
        this->removalWorker = std::thread(&InterSensorManager::removalWorker, this);
        this->removalWorker.detach();
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

    pcl::PointCloud<pcl::PointXYZRGBL> InterSensorManager::getMergedCloud(bool consume) {

        pcl::PointCloud<pcl::PointXYZRGBL> tmp;

        {
            // wait for the mutex
            std::unique_lock<std::mutex> lock(this->cloudMutex);

            // wait for the condition variable
            this->cloudConditionVariable.wait(lock, [this] { return this->readyToConsume; });

            tmp = *(this->mergedCloud.getPointCloud());

            if(consume)
                this->readyToConsume = false;
        }

        // notify the next thread in queue to continue
        this->cloudConditionVariable.notify_one();

        return tmp;
    }

    void InterSensorManager::removePointsByLabel(const std::set<std::uint32_t>& labels) {


        {
            // lock access to the queue
            std::unique_lock lock(this->cloudsToRemoveMutex);

            // if the queue is going to get full, remove the most recent
            // pop from the head
            if(this->cloudsToRemove.size() == MAX_REMOVAL_WORKER_QUEUE_LEN - 1)
                this->cloudsToRemove.pop_front();

            // add the new batch to the queue
            this->cloudsToRemove.push_front(labels);
        }

        // notify the removal worker
        this->cloudsToRemoveCond.notify_one();
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

            this->pendingCloudsCond.wait(lock, [this]() {
                return this->pendingCloudsQueue.size() < MAX_WORKER_QUEUE_LEN;
            });

            // verify if the queue has space
            // if it doesn't, remove the oldest
            if (this->pendingCloudsQueue.size() == MAX_WORKER_QUEUE_LEN - 1) {
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

        if(this->streamManagers.count(topicName) != 0)
            return;

        std::unique_ptr<IntraSensorManager> newStreamManager = std::make_unique<IntraSensorManager>(topicName, maxAge);

        // set the point removing method as a callback when some pointcloud ages on the stream manager
        newStreamManager->setPointAgingCallback([this](const std::set<std::uint32_t>& labels) {
                this->removePointsByLabel(labels);
        });

        // add a pointcloud whenever the IntraSensorManager has one ready
        newStreamManager->setPointCloudReadyCallback([this](entities::StampedPointCloud cloud, std::string& sensorName) {
                this->addSensorPointCloud(cloud, sensorName);
        });

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
            instance->cloudsToRemoveCond.notify_all();

            // wait for workers
            for(auto& w : instance->workers) {
                w.join();
            }

            // wait for point removal worker
            instance->removalWorker.join();

            delete instance;
            instance = nullptr;
        }
    }

    double InterSensorManager::getAverageRegistrationTime() {
        double val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->avgRegistrationTimeMs;
        }

        return val;
    }

    double InterSensorManager::getVarianceRegistrationTime() {
        double val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->varRegistrationTimeMs;
        }

        return val;
    }

    double InterSensorManager::getStdDevRegistrationTime() {

        double val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            val = std::sqrt(this->varRegistrationTimeMs);
        }

        return val;
    }

    size_t InterSensorManager::getSampleCount() {
        size_t val;

        {
            // acquire the mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->registrationTimeSampleCount;
        }

        return val;
    }

    double InterSensorManager::getIntraSensorAverageLatency() {

        double val;

        // TODO

        return val;
    }

    double InterSensorManager::getIntraSensorStdDev() {

        double val;

        // TODO

        return val;
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

            // notify next waiting thread
            this->pendingCloudsCond.notify_one();

            {
                // lock the point cloud mutex
                std::unique_lock lock(this->cloudMutex);

                // wait for the point cloud condition variable
                this->cloudConditionVariable.wait(lock, [this]() {
                    return !(this->workerProcessing) && !(this->beingExpired);
                });

                this->workerProcessing = true;

                // register the point cloud
                this->mergedCloud.registerPointCloud(newCloud.getPointCloud());

                this->workerProcessing = false;
                this->readyToConsume = true;
            }

            // std::cout << "[INTER] Registered new point cloud" << std::endl;

            // notify next waiting thread
            this->cloudConditionVariable.notify_one();

            // update the statistics
            auto now = utils::Utils::getCurrentTimeMillis();
            unsigned long long diff = now - newCloud.getTimestamp();
            {
                // acquire the mutex
                std::unique_lock lock(this->statisticsMutex);

                // update the statistics
                double delta = (double) diff - this->avgRegistrationTimeMs;
                (this->registrationTimeSampleCount)++;

                this->avgRegistrationTimeMs =
                        this->avgRegistrationTimeMs + delta / (double) this->registrationTimeSampleCount;
                if(this->registrationTimeSampleCount >= 2)
                    this->varRegistrationTimeMs =
                            this->varRegistrationTimeMs + delta * ((double) diff - this->avgRegistrationTimeMs);
            }
        }

    }

    void InterSensorManager::removalWorkerLoop() {

        /*

        std::set<std::uint32_t> labelsToRemove;

        while(true) {

            {
                // acquire the queue mutex
                std::unique_lock lock(this->cloudsToRemoveMutex);

                // wait for the condition variable
                // wait to have at least one point cloud to remove
                this->cloudsToRemoveCond.wait(lock, [this]() {
                    return this->cloudsToRemove.size() > 0 || this->workersShouldStop;
                });

                if(this->workersShouldStop)
                    return;

                // pick one batch from the queue
                // FIFO queue: pick from the tail
                labelsToRemove = this->cloudsToRemove.back();
                this->cloudsToRemove.pop_back(); // remove from the queue
            }

            // notify threads waiting to access the removal queue
            this->cloudsToRemoveCond.notify_one();

            {
                // acquire the point cloud mutex
                std::unique_lock lock(this->cloudMutex);

                // wait for the condition variable
                // nor the workers or age watchers can be manipulating
                this->cloudConditionVariable.wait(lock, [this]() {
                    return !(this->workerProcessing) && !(this->beingExpired);
                });

                // mark the point cloud as being expired
                this->beingExpired = true;

                // remove the points with the labels
                this->mergedCloud.removePointsWithLabels(labelsToRemove);

                std::cout << "[INTER] Removed points (" << labelsToRemove.size() << " clouds)" << std::endl;

                // clear the batch
                labelsToRemove.clear();

                // a new point cloud version is ready to be consumed
                this->beingExpired = false;
                this->readyToConsume = true;
            }

            // notify threads waiting to manipulate the point clouds
            this->cloudConditionVariable.notify_one();
        }

        */
    }
} // pcl_aggregator::managers
