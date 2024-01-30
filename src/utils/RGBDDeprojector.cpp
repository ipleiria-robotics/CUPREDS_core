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

#include <pcl_aggregator_core/utils/RGBDDeprojector.h>

namespace pcl_aggregator {
    namespace utils {
        RGBDDeprojector::RGBDDeprojector() {
            this->cloud = pcl::PointCloud<pcl::PointXYZRGBL>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBL>());
        }

        RGBDDeprojector::~RGBDDeprojector() {
            this->cloud.reset();
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

            // can't do anything without a depth image
            if(!this->isDepthImageSet)
                return;

            // TODO: create and call a CUDA kernel to deproject valid pixels into points
        }


    } // pcl_aggregator
} // utils