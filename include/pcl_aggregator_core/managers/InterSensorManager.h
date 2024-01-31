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

#ifndef CUPREDS_CORE_INTERSENSORMANAGER_H
#define CUPREDS_CORE_INTERSENSORMANAGER_H

#include <memory>
#include <unordered_map>
#include <cstddef>
#include <mutex>
#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.h>
#include <pcl_aggregator_core/managers/IntraSensorManager.h>
#include <pcl_aggregator_core/entities/StampedPointCloud.h>

#define GLOBAL_ICP_MAX_CORRESPONDENCE_DISTANCE 1.0f
#define GLOBAL_ICP_MAX_ITERATIONS 3

#define MAX_WORKER_QUEUE_LEN 10

#define VOXEL_LEAF_SIZE 0.2f

#define NUM_INTER_SENSOR_WORKERS 2

namespace pcl_aggregator::managers {

    struct pending_cloud_entry_t {
        entities::StampedPointCloud cloud;
        std::string sensorName;
        size_t queueIndex; // index in the queue
    };

    /*!
     * \brief Manage PointClouds coming from several sensors, like several LiDARs and depth cameras.
     *
     * This class keeps track of different sensors indexed by topic name, doing registration and point aging.
     * The IntraSensorManager class manages each individual sensor.
     * The instance of this class (only 1 because is a singleton) is a point clouds consumer.
     */
    class InterSensorManager {
        private:
            /*! \brief Singleton class instance. */
            static InterSensorManager* instance;
            /*! \brief The number of sensors being currently managed. */
            size_t nSources;
            /*! \brief The configured maximum point age. */
            double maxAge;
            /*! \brief Configured max memory to be consumed by a PointCloud in MB. */
            size_t maxMemory;
            /*! \brief Configured publish rate and registration rate. */
            size_t publishRate;
            /*! \brief Hash map of managers, one for each sensor (topic). */
            std::unordered_map<std::string,std::unique_ptr<IntraSensorManager>> streamManagers;
            /*! \brief Smart pointer to the merged PointCloud. */
            entities::StampedPointCloud mergedCloud;

            /*! \brief Mutex which manages concurrent access to the managers hash map. */
            std::mutex managersMutex;

            /*! \brief Mutex which manager concurrent access to the merged PointCloud pointer. */
            std::mutex cloudMutex;

            /*! \brief Condition variable to protect access to the merged PointCloud pointer. */
            std::condition_variable cloudConditionVariable;

            /*! \brief Is the merged PointCloud ready to be accessed. */
            bool cloudReady = true;

            /*! \brief Thread which monitors the PointCloud's memory usage. */
            std::thread memoryMonitoringThread;

            /*!\brief Flag to determine if the thread should be stopped or not. */
            bool keepThreadAlive = true;

            std::vector<std::thread> workers;

            /*! \brief Flag to signal workers to stop. */
            bool workersShouldStop = false;

            /*! \brief A queue with the sensor point clouds pending to be registered.
             *
             * Only the most recent for each sensor can be present here, at most.
             */
            std::deque<std::shared_ptr<struct pending_cloud_entry_t>> pendingCloudsQueue;

            /*! \brief A map of the pending sensor point clouds indexed by sensor name. */
            std::unordered_map<std::string,std::shared_ptr<pending_cloud_entry_t>> pendingCloudsBySensorName;

            /*! \brief Mutex to control access to the queue and map of pending point clouds. */
            std::mutex pendingCloudsMutex;

            /*! \brief Condition variable to constrol access to the queue and map of pending point clouds. */
            std::condition_variable pendingCloudsCond;

            /*! \brief InterSensorManager constructor.
             *
             * @param nSources Number of sensors to manage.
             * @param maxAge Maximum age for each point from the moment it is captured.
             * @param maxMemory Memory hard-limit for point clouds.
             * @param publishRate Point cloud publish rate in Hz.
             */
            InterSensorManager(size_t nSources, double maxAge, size_t maxMemory, size_t publishRate);

            /*! \brief InterSensorManager destructor. Destroys the IntraSensorManager instances. */
            ~InterSensorManager();

            /*! \brief Clear the points of the merged PointCloud. */
            void clearMergedCloud();

            /*! \brief Initialize the stream manager for a given topic with a given max age.
             *
             * @param topicName The name of the topic this IntraSensorManager will manage.
             * @param maxAge The point's maximum age for this sensor.
             */
            void initStreamManager(const std::string& topicName, double maxAge);

            /*! \brief Remove points with a given label from the merged PointCloud.
             * Used typically when points age, as a callback from the StreamManagers.
             *
             * @param label The label to remove.
             */
            void removePointsByLabel(const std::set<std::uint32_t>& labels);

            /*! \brief Add the processed PointCloud of a given sensor to the merged.
             * Used typically when the sensor finishes processing a new PointCloud.
             *
             * @param cloud The PointCloud to add.
             */
            void addSensorPointCloud(entities::StampedPointCloud cloud, std::string& sensorName);

        public:

            /*! \brief Singleton instance getter. */
            static InterSensorManager& get(size_t nSources, double maxAge, size_t maxMemory, size_t publishRate);

            /*! \brief Singleton instance destructor. */
            static void destruct();

            InterSensorManager(const InterSensorManager&) = delete; // delete the copy constructor

            InterSensorManager& operator=(const InterSensorManager&) = delete; // delete the assignment operator

            /*! \brief Get the number of sensors/streams being managed. */
            size_t getNClouds() const;

            /*! \brief Add a new PointCloud, coming from a given topic. The entrypoint of everything.
             *
             * @param cloud Smart pointer to the new pointcloud.
             * @param topicName The name of the topic from which the pointcloud came from. Will be used for identification.
             */
            void addCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const std::string& topicName);

            /*! \brief Set the transform of a given sensor, identified by the topic name, to the robot base frame.
             *
             * @param transform The affine transform between the sensor frame and the robot base frame.
             * @param topicName The name of the topic of the sensor with that transform.
             */
            void setTransform(const Eigen::Affine3d& transform, const std::string& topicName);

            pcl::PointCloud<pcl::PointXYZRGBL> getMergedCloud();

            void workersLoop();

        /*! \brief Memory monitoring routine.
         *
         * When a PointCloud reaches the defined max size, some points are removed. It runs contantly on a thread.
         *
         * @param instance Pointer to the InterSensorManager instance.
         * */
        friend void memoryMonitoringRoutine(InterSensorManager* instance);

        /*! \brief Routine to query the last pointcloud of a IntraSensorManager and register it.
         *
         * \param instance Pointer to the InterSensorManager instance.
         * \param topicName Name of the IntraSensorManager topic.
         */
        friend void streamCloudQuerierRoutine(InterSensorManager* instance, const std::string& topicName);

    };
} // pcl_aggregator::managers

#endif //CUPREDS_CORE_INTERSENSORMANAGER_H
