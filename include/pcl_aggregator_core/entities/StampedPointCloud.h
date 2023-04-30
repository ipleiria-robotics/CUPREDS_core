//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H
#define PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <eigen3/Eigen/Dense>
#include <cstdint>

#define POINTCLOUD_ORIGIN_NONE "none"

namespace pcl_aggregator {
    namespace entities {

        template <typename PointTypeT>
        class StampedPointCloud {

            private:
                unsigned long long timestamp;

                typename pcl::PointCloud<PointTypeT>::Ptr cloud = nullptr;
                bool transformComputed = false; // is the transform to the robot frame computed?
                bool icpTransformComputed = false; // is the transform computed by ICP computed?

                // the name of the topic the pointcloud came from
                std::string originTopic = POINTCLOUD_ORIGIN_NONE;

                std::uint32_t label; // label used to identify each pointcloud on removal
                std::uint32_t generateLabel();

            public:
                StampedPointCloud(std::string originTopic);

                unsigned long long getTimestamp() const;
                typename pcl::PointCloud<PointTypeT>::Ptr getPointCloud() const;
                std::string getOriginTopic() const;
                std::uint32_t getLabel() const;
                bool isIcpTransformComputed() const;
                void setTimestamp(unsigned long long t);
                void setPointCloud(typename pcl::PointCloud<PointTypeT>::Ptr cloud, bool assignGeneratedLabel=true);
                void setOriginTopic(std::string origin);

                bool isTransformComputed() const;
                void applyTransform(Eigen::Affine3d tf);


                void applyIcpTransform(Eigen::Matrix4f tf);

                void assignLabelToPointCloud(typename pcl::PointCloud<PointTypeT>::Ptr cloud, std::uint32_t label);
                void removePointsWithLabel(std::uint32_t label);

                template <typename RoutinePointTypeT>
                friend void transformPointCloudRoutine(StampedPointCloud<RoutinePointTypeT>* instance  );

                template <typename RoutinePointTypeT>
                friend void removePointsWithLabelRoutine(StampedPointCloud<RoutinePointTypeT>* instance, std::uint32_t label);

        };

        // custom comparison functor between stamped point clouds
        // they are compared by timestamp
        template <typename PointTypeT>
        struct CompareStampedPointCloudPointers {

            bool operator()(std::shared_ptr<StampedPointCloud<PointTypeT>> first, std::shared_ptr<StampedPointCloud<PointTypeT>> second) const {
                return first->getTimestamp() < second->getTimestamp();
            }
        };

    } // pcl_aggregator
} // entities

#endif //PCL_AGGREGATOR_CORE_STAMPEDPOINTCLOUD_H
