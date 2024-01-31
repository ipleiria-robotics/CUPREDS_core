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
#include "pcl_aggregator_core/cuda/CUDAPointClouds.cuh"


namespace pcl_aggregator::managers {

    void maxAgeWatchingRoutine(IntraSensorManager* instance) {

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

            // remove the points from this merged PointCloud
            std::thread pointCloudRemovalThread = std::thread(pointCloudRemovalRoutine, labelsToRemove);
            pthread_setname_np(pointCloudRemovalThread.native_handle(), "pointCloudRemovalThread");
            pointCloudRemovalThread.detach();

            // the point aging callback was set
            if(instance->pointAgingCallback != nullptr) {
                // call a thread to run the callback
                // if it was done in the same thread, it would delay the routine

                // remove the points from the InterSensorManager's merged pointcloud
                std::thread callbackThread = std::thread(instance->pointAgingCallback, labelsToRemove);
                callbackThread.detach();
            }

            // clear the labels set
            labelsToRemove.clear();

            // sleep for a second before repeating
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    IntraSensorManager::IntraSensorManager(const std::string& topicName, double maxAge) {
        this->topicName = topicName;
        this->cloud = std::make_unique<entities::StampedPointCloud>(topicName);
        this->maxAge = maxAge;

        // start the age watcher thread
        this->maxAgeWatcherThread = std::thread(maxAgeWatchingRoutine, this);
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

        for(const auto& c : this->clouds) {
            this->clouds.erase(c);
        }

        while(!this->cloudsNotTransformed.empty()) {
            this->cloudsNotTransformed.pop();
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
            std::lock_guard<std::mutex> cloudGuard(this->cloudMutex);

            // remove points with that label from the merged pointcloud
            this->cloud->removePointsWithLabel(label);
        }


        // lock the set
        std::lock_guard<std::mutex> guard(this->setMutex);

        // iterate the set
        for(auto& c : this->clouds) {
            if(c->getLabel() == label) {
                // remove the pointcloud from the set
                this->clouds.erase(c);
                break;
            }
        }

    }

    void IntraSensorManager::removePointClouds(std::set<std::uint32_t> labels) {

        {
            std::lock_guard<std::mutex> cloudGuard(this->cloudMutex);

            // remove points with that label from the merged pointcloud
            this->cloud->removePointsWithLabels(labels);
        }


        // lock the set
        std::lock_guard<std::mutex> guard(this->setMutex);

        // iterate the set
        for(auto& c : this->clouds) {
            if(labels.find(c->getLabel()) != labels.end()) {
                // remove the pointcloud from the set
                this->clouds.erase(c);
            }
        }

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

            // lock the mutex
            std::lock_guard<std::mutex> lock(this->cloudsNotRegisteredMutex);

            // if the queue is full, remove the oldest pointcloud
            // this producer doesn't wait for space, it just discards the oldest
            if(this->cloudsNotRegistered.size() == UNPROCESSED_CLOUD_MAX_QUEUE_LEN) {
                this->cloudsNotRegistered.pop_back();
            }

            // add the new pointcloud to the queue
            this->cloudsNotRegistered.push_front(std::move(spcl));
        }

        // notify a worker that a new pointcloud is available
        this->cloudsNotRegisteredCond.notify_one();

    }

    pcl::PointCloud<pcl::PointXYZRGBL> IntraSensorManager::getCloud() {

        pcl::PointCloud<pcl::PointXYZRGBL> result;

        {
            // create the unique_lock
            std::unique_lock lock(this->cloudMutex);

            // assign the value to the variable
            result = *this->cloud->getPointCloud();
        }

        // notify the next thread in queue
        this->cloudConditionVariable.notify_one();

        // return the pointcloud
        return result;
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

    std::function<void(std::set<std::uint32_t> labels)> IntraSensorManager::getPointAgingCallback() const {
        return this->pointAgingCallback;
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

        std::unique_ptr<entities::StampedPointCloud> cloudToRegister;

        while(true) {

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

            // apply the sensor transform to the new point cloud
            cloudToRegister->applyTransform(this->sensorTransform);

            entities::StampedPointCloud spcl(POINTCLOUD_ORIGIN_NONE);

            {
                // lock the cloud mutex
                std::unique_lock lock(this->cloudMutex);

                // register the point cloud
                this->cloud->registerPointCloud(cloudToRegister->getPointCloud());

                // get the point cloud
                spcl = *this->cloud;
            }

            // add the point cloud to the set
            {
                std::unique_lock lock(this->setMutex);
                // the points can now be removed, all that matters is the label from now on. saves memory
                cloudToRegister->getPointCloud()->clear();
                this->clouds.insert(std::move(cloudToRegister));
            }

            // call the InterSensorManager-defined callback
            // ATTENTION: on the InterSensorManager side this should be non-blocking, e.g., by adding to a queue of work
            this->pointCloudReadyCallback(spcl, this->topicName);

            // after completing the work, tell the next worker to pick a job
            this->cloudsNotRegisteredCond.notify_one();

        }
    }

} // pcl_aggregator::managers
