//
// Created by carlostojal on 01-05-2023.
//

#ifndef PCL_AGGREGATOR_CORE_RGBDDEPROJECTOR_H
#define PCL_AGGREGATOR_CORE_RGBDDEPROJECTOR_H

#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>

namespace pcl_aggregator {
    namespace utils {

        class RGBDDeprojector {

            private:
                Eigen::Matrix3d K; // camera intrinsic matrix
                bool isKSet = false;

                std::string camera_frame_id; // the frame id of the camera
                bool isFrameIdSet = false;

                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud; // the cloud generated by the deprojection

                cv::Mat last_depth_image;
                bool isDepthImageSet = false;

                cv::Mat last_color_image;
                bool isColorImageSet = false;

                /*!
                 * \brief Start deprojecting the images into the pointcloud of this instance.
                 * Does not need to receive or return anything, it just uses the instance properties.
                 */
                void deprojectImages();

            public:
                RGBDDeprojector();
                ~RGBDDeprojector();

                /*!
                 * \brief Get the camera intrinsic matrix.
                 * @return Camera intrinsic matrix
                 */
                Eigen::Matrix3d getK() const;
                /*!
                 * \brief Set the camera intrinsic matrix. Will be used to deproject the pixels into points.
                 * @param K
                 */
                void setK(const Eigen::Matrix3d& K);

                /*!
                 * \brief Get the set frame ID.
                 * @return String name of the frame ID.
                 */
                std::string getCameraFrameId() const;
                /*!
                 * \brief Set the camera frame ID. Will be used to transform the resulting points into the robot base frame.
                 * @param frame_id The name of the frame ID of the camera.
                 */
                void setCameraFrameId(const std::string& frame_id);

                /*!
                 * \brief Add a depth image to the processing pipeline.
                 * @param img The image to process.
                 */
                void addDepthImage(cv::Mat img);
                /*!
                 * \brief Add a color image to the processing pipeline.
                 * @param img The image to process.
                 */
                void addColorImage(cv::Mat img);

                /*!
                 * \brief Get the resulting deprojected colored pointcloud.
                 * @return The pointcloud generated from latest the color and depth image.
                 */
                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr getPointCloud() const;
        };

    } // pcl_aggregator
} // utils

#endif //PCL_AGGREGATOR_CORE_RGBDDEPROJECTOR_H
