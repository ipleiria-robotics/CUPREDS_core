//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_CUDA_POINTCLOUDS_H
#define PCL_AGGREGATOR_CORE_CUDA_POINTCLOUDS_H

#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>

namespace pcl_aggregator {
    namespace cuda {
        namespace pointclouds {

            /*! \brief Set a label to all PointCloud's points.
             *
             * @param cloud The PointCloud smart pointer to assign the label to.
             * @param label The 32-bit unsigned integer label.
             * */
            __host__ static void
            setPointCloudLabelCuda(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label);

            /*! \brief Transform all the points of a PointCloud using an affine transformation.
             *
             * @param transform The affine transform to apply.
             */
            __host__ static void
            transformPointCloudCuda(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr, Eigen::Affine3d transform);

            /*! \param The kernel which sets the label on an individual point.
             *
             * @param points An array of points got from the PointCloud.
             * @param label The label to assign.
             * @param num_points The point count (number of elements of the array "points").
             */
            __global__ void setPointLabelKernel(pcl::PointXYZRGBL *points, std::uint32_t label, int num_points);

            /*! \param The kernel which transforms a given point.
             *
             * @param points An array of points got from the PointCloud.
             * @param transform The transform to apply in homogenous coordinates (4x4 matrix of rotation and translation).
             * @param num_points The number of elements of the "points" array.
             */
            __global__ void transformPointKernel(pcl::PointXYZRGBL *points, Eigen::Matrix4d transform, int num_points);

        }
    } // pcl_aggregator
} // cuda

#endif //PCL_AGGREGATOR_CORE_CUDA_POINTCLOUDS_H
