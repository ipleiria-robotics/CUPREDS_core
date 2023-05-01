//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_CUDA_RGBD_CUH
#define PCL_AGGREGATOR_CORE_CUDA_RGBD_CUH

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cstdlib>

namespace pcl_aggregator {
    namespace cuda {
        namespace rgbd {

            /*! \brief Use CUDA to parallelize the deprojection of a depth image to a colored pointcloud.
             *
             * The points will be rejected if they surpass the depth limits imposed by minDepth and maxDepth.
             *
             * @param colorImage Color image as OpenCV matrix BGR8.
             * @param depthImage Depth image as OpenCV matrix.
             * @param K Camera intrinsic matrix as Eigen matrix.
             * @param minDepth Minimum admissible depth. Check the camera's datasheet.
             * @param maxDepth Maximum admissible depth. Check the camera's datasheet.
             * @param destination
             */
            __host__ void deprojectImages(const cv::Mat& colorImage, const cv::Mat& depthImage, const Eigen::Matrix3d& K,
                                          double minDepth, double maxDepth,
                                          pcl::PointCloud<pcl::PointXYZRGBL>::Ptr destination);


            /*! \brief The deprojection kernel, responsible by deprojecting an individual pixel.
             *
             * All pointers passed here must be in device memory.
             *
             * @param colorImage Color image in raw format in OpenCV ordering (row-major) BGR8.
             * @param depthImage Depth image in raw format in OpenCV ordering (row-major).
             * @param width Width of the images in pixels.
             * @param height Height of the images in pixels.
             * @param K Camera's intrinsic matrix.
             * @param minDepth Minimum admissible depth.
             * @param maxDepth Maximum admissible depth.
             * @param pointArray Array of colored points.
             * @param nValidPoints Pointer shared among threads. Each threads knows how many points have been inserted, thus inserting the next one immediately after.
             */
            __global__ void deprojectImagesKernel(unsigned char *colorImage, unsigned char *depthImage, unsigned int width,
                                                  unsigned int height, const Eigen::Matrix3d& K_inv,
                                                  double minDepth, double maxDepth, pcl::PointXYZRGBL *pointArray,
                                                  unsigned long long *nValidPoints);
        }
    } // pcl_aggregator
} // cuda

#endif //PCL_AGGREGATOR_CORE_CUDA_RGBD_CUH
