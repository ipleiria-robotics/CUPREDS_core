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

#include <pcl_aggregator_core/managers/IntraSensorManager.h>
#include "pcl_aggregator_core/cuda/CUDAPointClouds.h"


namespace pcl_aggregator::managers {

    IntraSensorManager::IntraSensorManager(const std::string& topicName, double maxAge) {
        this->topicName = topicName;
        this->cloud = std::make_unique<entities::StampedPointCloud>(topicName);
        this->maxAge = maxAge;

        // start the age watcher thread
        this->maxAgeWatcherThread = std::thread(&IntraSensorManager::ageWatcherLoop, this);
        pthread_setname_np(this->maxAgeWatcherThread.native_handle(), "maxAgeWatcherThread");
        this->maxAgeWatcherThread.detach();

        // start the workers
        for(size_t i = 0; i < NUM_INTRA_SENSOR_WORKERS; i++) {
            this->workers.emplace_back(&IntraSensorManager::workersLoop, this);
        }
    }

    IntraSensorManager::~IntraSensorManager() {
        this->cloud.reset();

        // signal the watcher to stop
        this->keepAgeWatcherAlive = false;
        // wait for the watcher to end
        this->maxAgeWatcherThread.join();

        // empty the set
        for(const auto& c : this->clouds) {
            this->clouds.erase(c);
        }

        // empty the transform buffer
        while(!this->cloudsNotTransformed.empty()) {
            this->cloudsNotTransformed.pop();
        }

        // empty the work queue
        while(!this->cloudsNotRegistered.empty()) {
            this->cloudsNotRegistered.pop_back();
        }

        // signal registration workers to stop
        this->workersShouldStop = true;
        this->cloudsNotRegisteredCond.notify_all();
    }

    bool IntraSensorManager::operator==(const IntraSensorManager &other) const {
        return this->topicName == other.topicName;
    }

    void IntraSensorManager::moveTransformPendingToQueue() {
        std::lock_guard<std::mutex> lock(this->cloudQueueMutex);

        while(!this->cloudsNotTransformed.empty()) {

            // add the point cloud to the job queue
            this->cloudsNotRegistered.push_front(std::move(this->cloudsNotTransformed.front()));

            // remove from the transform queue
            this->cloudsNotTransformed.pop();
        }
    }

    void IntraSensorManager::removePointCloud(std::uint32_t label) {

        {
            std::unique_lock<std::mutex> lock(this->cloudMutex);

            // wait for condition variable
            this->cloudConditionVariable.wait(lock, [this]() {
                return this->cloudReady;
            });

            this->cloudReady = false;

            // remove points with that label from the merged pointcloud
            // this->cloud->removePointsWithLabel(label);

            // iterate the set
            for (auto &c: this->clouds) {
                if (c->getLabel() == label) {
                    // remove the pointcloud from the set
                    this->clouds.erase(c);
                    break;
                }
            }

            this->cloudReady = true;
        }

        this->cloudConditionVariable.notify_one();
    }

    std::set<std::uint32_t> IntraSensorManager::removeExpiredPointClouds() {

        std::set<std::uint32_t> labelsToRemove;

        {
            std::unique_lock lock(this->cloudMutex);

            this->cloudConditionVariable.wait(lock, [this]() {
                return this->cloudReady;
            });

            this->cloudReady = false;

            // iterate the set
            for (auto &c: this->clouds) {
                if (c->getTimestamp() <= utils::Utils::getMaxTimestampForAge(this->maxAge)) {

                    // signal for removal
                    labelsToRemove.insert(c->getLabel());

                } else {
                    break;
                }
            }

            std::erase_if(this->clouds, [this](auto const& c) {
                return c->getTimestamp() <= utils::Utils::getMaxTimestampForAge(this->maxAge);
            });

            // remove all points marked for removal
            // this->cloud->removePointsWithLabels(labelsToRemove);

            this->cloudReady = true;

        }

        this->cloudConditionVariable.notify_one();

        return labelsToRemove;

    }

    void IntraSensorManager::addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr newCloud) {
        // check the incoming pointcloud for null or empty
        if(newCloud == nullptr)
            return;
        if(newCloud->empty()) {
            newCloud.reset();
            return;
        }

        // create a stamped point newCloud object to keep this pointcloud
        std::unique_ptr<entities::StampedPointCloud> spcl =
                std::make_unique<entities::StampedPointCloud>(this->topicName);
        // the new pointcloud is moved to the StampedPointCloud
        spcl->setPointCloud(std::move(newCloud));

        // if the transform is not set, add to the queue of clouds waiting for transform
        if(!this->sensorTransformSet) {
            // add the pointcloud to the queue
            // the ownership is moved to the queue
            // lock access to the queue
            std::lock_guard<std::mutex> lock(this->cloudQueueMutex);
            this->cloudsNotTransformed.push(std::move(spcl));
            return;
        } else {
            // if the transform is set, add to the queue of clouds waiting to be processed by the workers

            {
                // lock the mutex
                std::unique_lock<std::mutex> lock(this->cloudsNotRegisteredMutex);

                this->cloudsNotRegisteredCond.wait(lock, [this]() {
                    return this->cloudsNotRegistered.size() < UNPROCESSED_CLOUD_MAX_QUEUE_LEN;
                });

                // if the queue will become full, remove the oldest pointcloud
                // this producer doesn't wait for space, it just discards the oldest
                if (this->cloudsNotRegistered.size() == UNPROCESSED_CLOUD_MAX_QUEUE_LEN - 1) {
                    this->cloudsNotRegistered.pop_back();
                }

                // add the new pointcloud to the queue
                this->cloudsNotRegistered.push_front(std::move(spcl));
            }

            this->cloudsNotRegisteredCond.notify_one();
        }

    }

    void IntraSensorManager::setSensorTransform(const Eigen::Affine3d &transform) {

        {
            std::lock_guard<std::mutex> lock(this->sensorTransformMutex);

            // set the new transform
            this->sensorTransform = transform;
            this->sensorTransformSet = true;
        }

        // move the point clouds pending transform to the workers queue
        this->moveTransformPendingToQueue();
    }

    double IntraSensorManager::getMaxAge() const {
        return this->maxAge;
    }

    void IntraSensorManager::setPointAgingCallback(const std::function<void(std::set<std::uint32_t>)>& func) {
        this->pointAgingCallback = func;
    }

    std::function<void(entities::StampedPointCloud cloud, std::string& topicName)>
    IntraSensorManager::getPointCloudReadyCallback() const {
        return this->pointCloudReadyCallback;
    }

    void IntraSensorManager::setPointCloudReadyCallback(
            const std::function<void(entities::StampedPointCloud, std::string&)> &func) {
        this->pointCloudReadyCallback = func;
    }

    void IntraSensorManager::workersLoop() {

        while(true) {

            std::unique_ptr<entities::StampedPointCloud> cloudToRegister;

            {
                // acquire the queue mutex
                std::unique_lock lock(this->cloudsNotRegisteredMutex);

                // wait for a job to be available / stop if signaled
                this->cloudsNotRegisteredCond.wait(lock, [this]() {
                    return !this->cloudsNotRegistered.empty() || this->workersShouldStop;
                });
                // if the workers should stop, do it
                if (this->workersShouldStop)
                    return;

                // pick a pointcloud from the queue. move its ownership to this worker
                cloudToRegister = std::move(this->cloudsNotRegistered.front());
                this->cloudsNotRegistered.pop_front();
            }
            // notify next thread waiting to manipulate the queue
            this->cloudsNotRegisteredCond.notify_one();

            // apply the sensor transform to the new point cloud
            cloudToRegister->applyTransform(this->sensorTransform);

            entities::StampedPointCloud spcl(cloudToRegister->getOriginTopic());

            // assign the timestamp of the incoming point cloud to this new
            spcl.setTimestamp(cloudToRegister->getTimestamp());

            {
                // lock the cloud mutex
                std::unique_lock lock(this->cloudMutex);

                this->cloudConditionVariable.wait(lock, [this]() {
                    return this->cloudReady;
                });

                this->cloudReady = false;

                // register the point cloud
                // this->cloud->registerPointCloud(cloudToRegister->getPointCloud());

                // get the point cloud
                spcl.setPointCloudValue(*(cloudToRegister->getPointCloud()));

                // clear the points, will be no longer needed
                cloudToRegister->getPointCloud()->clear();
              
                // finally add the point cloud to the set
                this->clouds.insert(std::move(cloudToRegister));

                this->cloudReady = true;
            }

            this->cloudConditionVariable.notify_one();

            // std::cout << "[INTRA] Processed new point cloud" << std::endl;

            // measure the time difference between capture and now
            auto now = utils::Utils::getCurrentTimeMillis();
            unsigned long long diff = now - spcl.getTimestamp();

            {
                // acquire the mutex
                std::unique_lock lock(this->statisticsMutex);

                // use welford's method to update the statistics
                double delta = diff - this->avgRegistrationTimeMs;
                this->avgRegistrationTimeMs += delta / (this->registrationTimeSampleCount + 1);
                this->meanSquaredRegistrationTimeMs += delta * (diff - this->avgRegistrationTimeMs);

                // update the variance
                if(this->registrationTimeSampleCount >= 2)
                    this->varRegistrationTimeMs = this->meanSquaredRegistrationTimeMs / (this->registrationTimeSampleCount - 1);

                // increment the sample count
                this->registrationTimeSampleCount++;
            }

            // call the InterSensorManager-defined callback
            // ATTENTION: on the InterSensorManager side this should be non-blocking, e.g., by adding to a queue of work
            this->pointCloudReadyCallback(spcl, this->topicName);

        }
    }

    void IntraSensorManager::ageWatcherLoop() {

        // set of labels with expired timestamps
        std::set<std::uint32_t> labelsToRemove;

        // while not signaled to stop
        while(this->keepAgeWatcherAlive) {

            // remove the point clouds
            labelsToRemove = this->removeExpiredPointClouds();

            if(labelsToRemove.empty())
                continue;

            // call the inter-sensor callback
            if(this->pointAgingCallback != nullptr) {
                this->pointAgingCallback(labelsToRemove);
            } else {
                throw std::runtime_error("Point ageing callback not set!");
            }

            // std::cout << "[INTRA] Removing points (" << labelsToRemove.size() << " clouds)" << std::endl;

            // clear the labels set
            labelsToRemove.clear();

            // sleep for the period
            std::this_thread::sleep_for(std::chrono::seconds(AGE_WATCHER_PERIOD_SECONDS));
        }
    }

    double IntraSensorManager::getAverageRegistrationTime() {
        double val;

        {
            // acquire mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->avgRegistrationTimeMs;
        }

        return val;
    }

    double IntraSensorManager::getVarianceRegistrationTime() {
        double val;

        {
            // acquire mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->varRegistrationTimeMs;
        }

        return val;
    }

    double IntraSensorManager::getStdDevRegistrationTime() {
        double val;

        {
            // acquire mutex
            std::unique_lock lock(this->statisticsMutex);

            val = std::sqrt(this->varRegistrationTimeMs);
        }

        return val;
    }

    size_t IntraSensorManager::getSampleCount() {
        size_t val;

        {
            // acquire mutex
            std::unique_lock lock(this->statisticsMutex);

            val = this->registrationTimeSampleCount;
        }

        return val;
    }

} // pcl_aggregator::managers
