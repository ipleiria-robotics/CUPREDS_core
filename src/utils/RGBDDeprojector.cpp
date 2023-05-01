//
// Created by carlostojal on 01-05-2023.
//

#include "pcl_aggregator_core/utils/RGBDDeprojector.h"

namespace pcl_aggregator {
    namespace utils {
        RGBDDeprojector::RGBDDeprojector() {
            // TODO: init the pointcloud
        }

        RGBDDeprojector::~RGBDDeprojector() {
            // TODO: destroy the pointcloud
        }

        Eigen::Matrix3d RGBDDeprojector::getK() const {
            if(!this->isKSet)
                throw std::runtime_error("The intrinsic matrix K was not set!");

            return this->K;
        }

        void RGBDDeprojector::setK(const Eigen::Matrix3d &K) {
            this->K = K;
            this->isKSet = true;
        }

        std::string RGBDDeprojector::getCameraFrameId() const {
            if(!this->isFrameIdSet)
                throw std::runtime_error("The camera frame ID was not set!");

            return this->camera_frame_id;
        }

        void RGBDDeprojector::setCameraFrameId(const std::string &frame_id) {
            this->camera_frame_id = frame_id;
            this->isFrameIdSet = true;
        }

        void RGBDDeprojector::addDepthImage(cv::Mat img) {
            this->last_depth_image = img;
            this->isDepthImageSet = true;
            // TODO: start processing image
        }

        void RGBDDeprojector::addColorImage(cv::Mat img) {
            this->last_color_image = img;
            this->isColorImageSet = true;
            // TODO: start processing image
        }

        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr RGBDDeprojector::getPointCloud() const {
            return this->cloud;
        }

        void RGBDDeprojector::deprojectImages() {
            // TODO: create and call a CUDA kernel to deproject valid pixels into points
        }


    } // pcl_aggregator
} // utils