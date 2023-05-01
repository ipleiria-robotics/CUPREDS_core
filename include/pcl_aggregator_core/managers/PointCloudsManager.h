//
// Created by carlostojal on 01-05-2023.
//

#ifndef PCL_AGGREGATOR_CORE_POINTCLOUDSMANAGER_H
#define PCL_AGGREGATOR_CORE_POINTCLOUDSMANAGER_H

#include <memory>
#include <unordered_map>
#include <cstddef>
#include <mutex>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_aggregator_core/managers/StreamManager.h>

#define GLOBAL_ICP_MAX_CORRESPONDENCE_DISTANCE 1
#define GLOBAL_ICP_MAX_ITERATIONS 10

namespace pcl_aggregator {
    namespace managers {

        /*!
         * \brief Manage PointClouds coming from several sensors, like several LiDARs and depth cameras.
         *
         * This class keeps track of different sensors indexed by topic name, doing registration and point aging.
         * The StreamManager class manages each individual sensor.
         */
        class PointCloudsManager {
            private:
                /*! \brief The number of sensors being currently managed. */
                size_t nSources;
                /*! \brief The configured maximum point age. */
                double maxAge;
                /*! \brief Hash map of managers, one for each sensor (topic). */
                std::unordered_map<std::string,std::unique_ptr<StreamManager>> streamManagers;
                /*! \brief Smart pointer to the merged PointCloud. */
                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr mergedCloud;

                /*! \brief Mutex which manages concurrent access to the managers hash map. */
                std::mutex managersMutex;

                /*! \brief Append the points of one PointCloud to the merged version of this manager.
                 *
                 * @param input The shared pointer to the input PointCloud.
                 * @return Flag denoting if ICP was possible or not.
                 */
                bool appendToMerged(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& input);

                /*! \brief Clear the points of the merged PointCloud. */
                void clearMergedCloud();

                /*! \brief Initialize the stream manager for a given topic with a given max age.
                 *
                 * @param topicName The name of the topic this StreamManager will manage.
                 * @param maxAge The point's maximum age for this sensor.
                 */
                void initStreamManager(const std::string& topicName, double maxAge);

            public:
                PointCloudsManager(size_t nSources, double maxAge);
                ~PointCloudsManager();

                /*! \brief Get the number of sensors/streams being managed. */
                size_t getNClouds() const;

                /*! \brief Add a new PointCloud, coming from a given topic. The entrypoint of everything.
                 *
                 * @param cloud Smart pointer to the new pointcloud.
                 * @param topicName The name of the topic from which the pointcloud came from. Will be used for identification.
                 */
                void addCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, const std::string& topicName);

                /*! \brief Set the transform of a given sensor, identified by the topic name, to the robot base frame.
                 *
                 * @param transform The affine transform between the sensor frame and the robot base frame.
                 * @param topicName The name of the topic of the sensor with that transform.
                 */
                void setTransform(const Eigen::Affine3d& transform, const std::string& topicName);

                pcl::PointCloud<pcl::PointXYZRGBL> getMergedCloud();

        };

    } // pcl_aggregator
} // managers

#endif //PCL_AGGREGATOR_CORE_POINTCLOUDSMANAGER_H
