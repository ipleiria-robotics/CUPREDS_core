//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H
#define PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <mutex>

#define POINTCLOUD_ORIGIN_NONE "none"

namespace pcl_aggregator {
    namespace entities {

        /*! \brief Stamped Point Cloud
         *         A PointCloud with an associated timestamp. Also has other utilities.
         */
        class StampedPointCloud {

            private:
                /*! \brief The timestamp */
                unsigned long long timestamp;

                /*! \brief The PointCloud */
                typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud = nullptr;

                /*! \brief Was the transform to the robot frame computed? */
                bool transformComputed = false;

                /*! \brief Was the transform computed by ICP applied? */
                bool icpTransformComputed = false;

                // the name of the topic the pointcloud came from
                std::string originTopic = POINTCLOUD_ORIGIN_NONE;

                /*! \brief Label used to identify each PointCloud, for example on removal */
                std::uint32_t label;

                /*! \brief Mutex to contain access to this PointCloud. */
                std::mutex cloudMutex;

                /*! \brief Generate a label to the PointCloud based on the origin topic name and timestamp. */
                std::uint32_t generateLabel();

            public:
                StampedPointCloud(std::string originTopic);
                ~StampedPointCloud();

                /*! \brief Get the PointCloud timestamp. */
                unsigned long long getTimestamp() const;
                /*! \brief Get a smart pointer to the PointCloud. */
                typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr getPointCloud();
                /*! \brief Get the origin topic name. */
                std::string getOriginTopic() const;
                /*! \brief Get the label of the PointCloud. Should be unique. */
                std::uint32_t getLabel() const;
                /*! \brief Set the PointCloud's timestamp. */
                void setTimestamp(unsigned long long t);
                /*! \brief Set the PointCloud by smart pointer. This method moves the pointer, does not copy it or increment use count.
                 * @param cloud The PointCloud smart pointer to move from.
                 * @param assignGeneratedLabel Assign a generated label or not. Generating the label has an additional overhead, but is usually needed.
                 * */
                void setPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, bool assignGeneratedLabel=true);
                /*! \brief Set the origin topic name.
                 * @param origin The topic name.
                 */
                void setOriginTopic(const std::string& origin);

                /*! \brief Check if the transform to the robot base frame was computed. */
                bool isTransformComputed() const;
                /*! \brief Apply the robot frame transform. */
                void applyTransform(const Eigen::Affine3d& tf);

                /*! \brief Check if the ICP transform was computed on this PointCloud. */
                bool isIcpTransformComputed() const;
                /*! \brief Apply the ICP transform. */
                void applyIcpTransform(const Eigen::Matrix4f& tf);

                /*! \brief Assign a label to a PointCloud.
                 *
                 * @param cloud The PointCloud's smart pointer.
                 * @param label The 32-bit unsigned label to assign.
                 */
                static void assignLabelToPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label);
                /*! \brief Remove points with a given label from the current PointCloud.
                 *
                 * @param label The label to remove.
                 */
                void removePointsWithLabel(std::uint32_t label);

        };

        /*! \brief Custom comparison functor between stamped point clouds. The comparison criteria is the timestamp. */
        struct CompareStampedPointCloudPointers {

            bool operator()(std::shared_ptr<StampedPointCloud> first, std::shared_ptr<StampedPointCloud> second) const {
                return first->getTimestamp() < second->getTimestamp();
            }
        };

    } // pcl_aggregator
} // entities

#endif //PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H
