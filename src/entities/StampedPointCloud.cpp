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

#include <pcl_aggregator_core/entities/StampedPointCloud.h>
#include <pcl_aggregator_core/utils/Utils.h>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>
#include <utility>


namespace pcl_aggregator::entities {

    StampedPointCloud::StampedPointCloud(const std::string& originTopic) {
        this->timestamp = utils::Utils::getCurrentTimeMillis();

        this->setOriginTopic(originTopic);

        this->label = generateLabel();

        this->cloud = pcl::PointCloud<pcl::PointXYZRGBL>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBL>());
    }

    StampedPointCloud::~StampedPointCloud() {

        // the StampedPointCloud owns its cloud's pointer and should destroy it
        this->cloud.reset();
    }

    // generate a 32-bit label and assign
    std::uint32_t StampedPointCloud::generateLabel() {

        std::string combined = this->originTopic + std::to_string(this->timestamp);

        std::hash<std::string> hasher;
        std::uint32_t hash_value = hasher(combined);

        return hash_value;
    }

    unsigned long long StampedPointCloud::getTimestamp() const {
        return this->timestamp;
    }

    typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& StampedPointCloud::getPointCloud() {
        return cloud;
    }

    std::string StampedPointCloud::getOriginTopic() const {
        return this->originTopic;
    }

    std::uint32_t StampedPointCloud::getLabel() const {
        return this->label;
    }

    void StampedPointCloud::setPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr c, bool assignGeneratedLabel) {

        // free the old pointcloud
        this->cloud.reset();

        // set the new
        this->cloud = std::move(c);

        if(this->cloud != nullptr) {
            if (assignGeneratedLabel)
                StampedPointCloud::assignLabelToPointCloud(this->cloud, this->label);
        } else {
            std::cerr << "StampedPointCloud::setPointCloud: cloud is null!" << std::endl;
        }
    }

    void StampedPointCloud::assignLabelToPointCloud(const typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, std::uint32_t label) {

        if(cloud != nullptr) {
            cuda::pointclouds::setPointCloudLabelCuda(cloud, label);
        } else {
            std::cerr << "StampedPointCloud::assignLabelToPointCloud: cloud is null!" << std::endl;
        }
    }

    void StampedPointCloud::setOriginTopic(const std::string& origin) {
        this->originTopic = origin;
    }

    bool StampedPointCloud::isTransformComputed() const {
        return this->transformComputed;
    }

    void StampedPointCloud::applyTransform(const Eigen::Affine3d& tf) {

        if(this->cloud != nullptr) {

            // call a CUDA thread to transform the pointcloud in-place
            cuda::pointclouds::transformPointCloudCuda(this->cloud, tf);
        } else {
            std::cerr << "StampedPointCloud::applyTransform: cloud is null!" << std::endl;
        }

        // pcl::transformPointCloud(*this->cloud, *this->cloud, tf);
        this->transformComputed = true;
    }

    void StampedPointCloud::removePointsWithLabel(std::uint32_t label) {

        auto it = this->cloud->begin();
        while (it != this->cloud->end()) {
            if (it->label == label)
                it = this->cloud->erase(it);
            else
                ++it;
        }
    }

    void StampedPointCloud::removePointsWithLabels(const std::set<std::uint32_t> &labels) {

        auto it = this->cloud->begin();
        while (it != this->cloud->end()) {
            // if the label is in the set, remove it
            if (labels.find(it->label) != labels.end())
                it = this->cloud->erase(it);
            else
                ++it;
        }
    }

    void StampedPointCloud::downsample(float leafSize) {

        pcl::VoxelGrid<pcl::PointXYZRGBL> voxelGrid;
        voxelGrid.setInputCloud(this->cloud);
        voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
        voxelGrid.filter(*this->cloud);
    }

    void StampedPointCloud::registerPointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& newCloud, bool thisAsCenter) {

        // if the new point cloud is empty, not need to further processing
        if(newCloud->empty())
            return;

        // if this point cloud is empty, it can just become the new
        if(this->cloud->empty()) {
            if (cuda::pointclouds::concatenatePointCloudsCuda(this->cloud,
                                                              reinterpret_cast<const pcl::PointCloud<pcl::PointXYZRGBL> &>(newCloud)) <
                0) {
                throw std::runtime_error("Error copying incoming point cloud");
            }
            return;
        }

        // if none of the point clouds are empty, do the registration
        pcl::IterativeClosestPoint<pcl::PointXYZRGBL,pcl::PointXYZRGBL> icp;

        // set ICP convergence parameters
        icp.setMaxCorrespondenceDistance(MAX_CORRESPONDENCE_DISTANCE);
        icp.setMaximumIterations(MAX_ICP_ITERATIONS);

        if(thisAsCenter) {
            // the incoming point cloud is transformed
            icp.setInputSource(newCloud);
            icp.setInputTarget(this->cloud);
            // transform newCloud
            icp.align(*newCloud);
        } else {
            // this point cloud is transformed
            icp.setInputSource(this->cloud);
            icp.setInputTarget(newCloud);
            // transform this->cloud
            icp.align(*this->cloud);
        }

        // merge the point clouds after registration
        if(cuda::pointclouds::concatenatePointCloudsCuda(this->cloud, reinterpret_cast<const pcl::PointCloud<pcl::PointXYZRGBL> &>(newCloud)) < 0) {
            throw std::runtime_error("Error concatenating point clouds after registration");
        }

        this->downsample(ICP_DOWNSAMPLE_SIZE);
    }
} // pcl_aggregator::entities