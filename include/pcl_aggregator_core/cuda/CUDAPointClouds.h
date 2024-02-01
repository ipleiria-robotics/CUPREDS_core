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
             *
             * @returns Zero on success, a negative number on error.
             * */
            __host__ int setPointCloudLabelCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, std::uint32_t label);

            /*! \brief Transform all the points of a PointCloud using an affine transformation.
             *
             * @param transform The affine transform to apply.
             *
             * @returns Zero on success, a negative number on error.
             */
            __host__ int transformPointCloudCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, const Eigen::Affine3d& transform);

            /*! \brief Concatenate the points of cloud2 into cloud1.
             *
             * @param cloud1 The PointCloud which will receive the points.
             * @param cloud2 The PointCloud which gives the points.
             *
             * @returns Zero on success, a negative number on error.
             */
            __host__ int concatenatePointCloudsCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud1,
                                                     const pcl::PointCloud<pcl::PointXYZRGBL>& cloud2);

            /*! \brief The kernel which sets the label on an individual point.
             *
             * @param points An array of points got from the PointCloud.
             * @param label The label to assign.
             * @param num_points The point count (number of elements of the array "points").
             *
             * @returns Zero on success, a negative number on error.
             */
            __global__ void setPointLabelKernel(pcl::PointXYZRGBL *points, std::uint32_t label, int num_points);

            /*! \brief The kernel which transforms a given point.
             *
             * @param points An array of points got from the PointCloud.
             * @param transform The transform to apply in homogenous coordinates (4x4 matrix of rotation and translation).
             * @param num_points The number of elements of the "points" array.
             *
             * @returns Zero on success, a negative number on error.
             */
            __global__ void transformPointKernel(pcl::PointXYZRGBL *points, Eigen::Matrix4d transform, int num_points);

            /*! \brief The kernel which concatenates a point on cloud2 to cloud1.
             *
             * @param cloud1 Array of points of the cloud1.
             * @param cloud1_original_size The size of cloud1 before concatenation.
             * @param cloud2 Array of points of the cloud2.
             * @param cloud2_size The size of cloud2.
             *
             * @returns Zero on success, a negative number on error.
             */
            __global__ void concatenatePointCloudsKernel(pcl::PointXYZRGBL* cloud1, std::size_t cloud1_original_size,
                                                         pcl::PointXYZRGBL* cloud2, std::size_t cloud2_size);

        }
    } // pcl_aggregator
} // cuda

#endif //PCL_AGGREGATOR_CORE_CUDA_POINTCLOUDS_H
