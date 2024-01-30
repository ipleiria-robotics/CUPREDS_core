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
        this->cloud = std::make_shared<entities::StampedPointCloud>(topicName);
        this->maxAge = maxAge;

        // start the age watcher thread
        this->maxAgeWatcherThread = std::thread(maxAgeWatchingRoutine, this);
        pthread_setname_np(this->maxAgeWatcherThread.native_handle(), "maxAgeWatcherThread");

        // this thread can detach from the main thread
        this->maxAgeWatcherThread.detach();
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
    }

    bool IntraSensorManager::operator==(const IntraSensorManager &other) const {
        return this->topicName == other.topicName;
    }

    void IntraSensorManager::computeTransform() {
        while(!this->cloudsNotTransformed.empty()) {

            // get the first element
            std::shared_ptr<entities::StampedPointCloud> spcl = this->cloudsNotTransformed.front();
            spcl->applyTransform(this->sensorTransform);

            // add to the set
            this->clouds.insert(std::move(spcl));

            // remove from the queue
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
        for(auto c : this->clouds) {
            if(c->getLabel() == label) {
                // remove the pointcloud from the set
                this->clouds.erase(c);
                c.reset();
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
        for(auto c : this->clouds) {
            if(labels.find(c->getLabel()) != labels.end()) {
                // remove the pointcloud from the set
                this->clouds.erase(c);
                // free the pointcloud pointer
                c.reset();
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

        if(!this->sensorTransformSet) {
            // add the pointcloud to the queue
            // the ownership is moved to the queue
            // lock access to the queue
            std::lock_guard<std::mutex> lock(this->cloudQueueMutex);
            this->cloudsNotTransformed.push(std::move(spcl));
            return;
        }

        // transform the incoming pointcloud and add directly to the set

        // start a thread to transform the pointcloud
        auto transformRoutine = [this] (const std::unique_ptr<entities::StampedPointCloud>& spcl, const Eigen::Affine3d& tf) {
            applyTransformRoutine(this, spcl, tf);
        };

        /*
        // pointcloud is passed as a const reference: ownership is not moved and no copy is made
        std::thread transformationThread(transformRoutine, std::ref(spcl), sensorTransform);

        // wait for the thread
        transformationThread.join();*/

        transformRoutine(std::ref(spcl), sensorTransform);

        try {
            if(!spcl->getPointCloud()->empty()) {

                {
                    // lock the pointcloud mutex
                    std::unique_lock<std::mutex> lock(this->cloudMutex);

                    // set that the pointcloud is being registered
                    this->cloudReady = false;

                    if (!this->cloud->getPointCloud()->empty()) {

                        pcl::IterativeClosestPoint<pcl::PointXYZRGBL,pcl::PointXYZRGBL> icp;

                        // align with the most recent point cloud in the origin of the frame (robot-centric)
                        icp.setInputSource(this->cloud->getPointCloud());
                        icp.setInputTarget(spcl->getPointCloud());

                        icp.setMaxCorrespondenceDistance(STREAM_ICP_MAX_CORRESPONDENCE_DISTANCE);
                        icp.setMaximumIterations(STREAM_ICP_MAX_ITERATIONS);

                        icp.align(*this->cloud->getPointCloud());
                    }

                    if (cuda::pointclouds::concatenatePointCloudsCuda(this->cloud->getPointCloud(),
                                                                      *(spcl->getPointCloud())) < 0) {
                        std::cerr << "Could not concatenate the pointclouds at the IntraSensorManager!" << std::endl;
                    }

                    // downsample the new merged pointcloud
                    this->cloud->downsample(STREAM_DOWNSAMPLING_LEAF_SIZE);

                    // set the registration as finished
                    this->cloudReady = true;
                }

                // by this time, the mutex is out of scope, thus fred

                // notify the waiting threads after releasing the mutex
                this->cloudConditionVariable.notify_one();

                // the points are no longer needed
                spcl->getPointCloud()->clear();

                /*
                if(this->pointCloudReadyCallback != nullptr) {

                    std::lock_guard<std::mutex> cloudGuard1(this->cloudMutex);

                    // call the callback on a new thread
                    std::thread pointCloudCallbackThread = std::thread([this]() {
                        this->pointCloudReadyCallback(std::ref(this->cloud->getPointCloud()),std::ref(this->cloudMutex));
                    });
                    pointCloudCallbackThread.detach();
                }*/
            }

        } catch (std::exception &e) {
            std::cerr << "Error performing sensor-wise ICP: " << e.what() << std::endl;
        }

    }

    pcl::PointCloud<pcl::PointXYZRGBL> IntraSensorManager::getCloud() {

        pcl::PointCloud<pcl::PointXYZRGBL> result;

        {
            // create the unique_lock
            std::unique_lock lock(this->cloudMutex);

            // wait for cloudReady to be "true" and to be notified
            // if the producer is not currently working, this would wait indefinitely without returning any point cloud
            // hence using "wait_for": to unblock execution in that situation
            this->cloudConditionVariable.wait_for(lock, std::chrono::milliseconds(CONSUMER_TIMEOUT_MS),
                                                  [this] { return this->cloudReady; });

            // assign the value to the variable
            result = *this->cloud->getPointCloud();
        }

        // notify the next thread in queue
        this->cloudConditionVariable.notify_one();

        // return the pointcloud
        return result;
    }

    void IntraSensorManager::setSensorTransform(const Eigen::Affine3d &transform) {

        std::lock_guard<std::mutex> lock(this->sensorTransformMutex);

        // set the new transform
        this->sensorTransform = transform;
        this->sensorTransformSet = true;
        this->computeTransform();
    }

    double IntraSensorManager::getMaxAge() const {
        return this->maxAge;
    }

    void applyTransformRoutine(IntraSensorManager *instance,
                               const std::unique_ptr<entities::StampedPointCloud>& spcl,
                               const Eigen::Affine3d& tf) {
        spcl->applyTransform(tf);
    }

    std::function<void(std::set<std::uint32_t> labels)> IntraSensorManager::getPointAgingCallback() const {
        return this->pointAgingCallback;
    }

    void IntraSensorManager::setPointAgingCallback(const std::function<void(std::set<std::uint32_t>)>& func) {
        this->pointAgingCallback = func;
    }

    std::function<void(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud, std::mutex& cloudMutex)>
    IntraSensorManager::getPointCloudReadyCallback() const {
        return this->pointCloudReadyCallback;
    }

    void IntraSensorManager::setPointCloudReadyCallback(
            const std::function<void(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &, std::mutex&)> &func) {
        this->pointCloudReadyCallback = func;
    }

} // pcl_aggregator::managers
