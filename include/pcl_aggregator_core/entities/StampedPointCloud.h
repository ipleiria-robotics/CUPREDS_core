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

#ifndef CUPREDS_CORE_STAMPEDPOINTCLOUD_H
#define CUPREDS_CORE_STAMPEDPOINTCLOUD_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <eigen3/Eigen/Dense>
#include <cstdint>
#include <set>
#include <mutex>

#define POINTCLOUD_ORIGIN_NONE "none"

#define MAX_CORRESPONDENCE_DISTANCE 0.1f
#define MAX_ICP_ITERATIONS 10
#define ICP_DOWNSAMPLE_SIZE 0.1f

#define NDT_TRANSFORMATION_EPSILON 0.01f
#define NDT_STEP_SIZE 0.1f
#define NDT_RESOLUTION 1.0f
#define NDT_MAX_ITERATIONS 10

namespace pcl_aggregator::entities {

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

            // the name of the topic the pointcloud came from
            std::string originTopic = POINTCLOUD_ORIGIN_NONE;

            /*! \brief Label used to identify each PointCloud, for example on removal */
            std::uint32_t label;


            /*! \brief Generate a label to the PointCloud based on the origin topic name and timestamp. */
            std::uint32_t generateLabel();

        public:
            StampedPointCloud(const std::string& originTopic);
            ~StampedPointCloud();

            bool operator==(const StampedPointCloud& other);
            bool operator==(const std::unique_ptr<StampedPointCloud>& other);

            /*! \brief Get the PointCloud timestamp. */
            unsigned long long getTimestamp() const;

            /*! \brief Assign a new timestamp to the PointCloud. */
            void setTimestamp(unsigned long long timestamp);

            /*! \brief Get a smart pointer to the PointCloud. */
            typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& getPointCloud();

            /*! \brief Get the origin topic name. */
            std::string getOriginTopic() const;

            /*! \brief Get the label of the PointCloud. Should be unique. */
            std::uint32_t getLabel() const;

            /*! \brief Set the PointCloud by smart pointer. This method moves the pointer, does not copy it or increment use count.
             * @param cloud The PointCloud smart pointer to move from.
             * @param assignGeneratedLabel Assign a generated label or not. Generating the label has an additional overhead, but is usually needed.
             * */
            void setPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, bool assignGeneratedLabel=true);

            void setPointCloudValue(pcl::PointCloud<pcl::PointXYZRGBL> cloud, bool assignGeneratedLabel=true);

            /*! \brief Set the origin topic name.
             * @param origin The topic name.
             */
            void setOriginTopic(const std::string& origin);

            /*! \brief Check if the transform to the robot base frame was computed. */
            bool isTransformComputed() const;

            /*! \brief Apply the robot frame transform. */
            void applyTransform(const Eigen::Affine3d& tf);

            /*! \brief Assign a label to a PointCloud.
             *
             * @param cloud The PointCloud's smart pointer.
             * @param label The 32-bit unsigned label to assign.
             */
            static void assignLabelToPointCloud(const typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, std::uint32_t label);

            /*! \brief Remove points with a given label from the current PointCloud.
             *
             * @param label The label to remove.
             */
            void removePointsWithLabel(std::uint32_t label);

            /*! \brief Remove points with the given labels from the current PointCloud.
             *
             * @param labels The labels to remove.
             */
            void removePointsWithLabels(const std::set<std::uint32_t> &labels);

            /*! \brief Apply voxel grid filter to the PointCloud.
             *
             * @param leafSize
             */
            void downsample(float leafSize);

            /*! \brief Register another PointCloud.
             *
             * @param cloud The PointCloud to register.
             * @param thisAsCenter Use this instance as the center of the frame, or the oncoming?
             */
            void registerPointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, bool thisAsCenter = false);

    };

    /*! \brief Custom comparison functor between stamped point clouds. The comparison criteria is the timestamp. */
    struct CompareStampedPointCloudPointers {

        bool operator()(const std::unique_ptr<StampedPointCloud>& first,
                const std::unique_ptr<StampedPointCloud>& second) const {
            return first->getTimestamp() < second->getTimestamp();
        }
    };

} // pcl_aggregator::entities

#endif //CUPREDS_CORE_STAMPEDPOINTCLOUD_H
