//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/entities/StampedPointCloud.h>
#include <pcl_aggregator_core/utils/Utils.h>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>
#include <utility>

namespace pcl_aggregator {
    namespace entities {

        StampedPointCloud::StampedPointCloud(std::string originTopic) {
            this->timestamp = utils::Utils::getCurrentTimeMillis();

            this->setOriginTopic(originTopic);

            this->label = generateLabel();

            this->cloud = pcl::PointCloud<pcl::PointXYZRGBL>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBL>());
        }

        StampedPointCloud::~StampedPointCloud() {

            std::lock_guard<std::mutex> lock(cloudMutex);
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
            std::lock_guard<std::mutex> lock(cloudMutex);
            return cloud;
        }

        std::string StampedPointCloud::getOriginTopic() const {
            return this->originTopic;
        }

        std::uint32_t StampedPointCloud::getLabel() const {
            return this->label;
        }

        bool StampedPointCloud::isIcpTransformComputed() const {
            return icpTransformComputed;
        }

        void StampedPointCloud::setTimestamp(unsigned long long t) {
            this->timestamp = t;
        }

        void StampedPointCloud::setPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr c, bool assignGeneratedLabel) {

            std::lock_guard<std::mutex> lock(cloudMutex);

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

            std::lock_guard<std::mutex> lock(cloudMutex);

            if(this->cloud != nullptr) {

                // call a CUDA thread to transform the pointcloud in-place
                cuda::pointclouds::transformPointCloudCuda(this->cloud, tf);
            } else {
                std::cerr << "StampedPointCloud::applyTransform: cloud is null!" << std::endl;
            }

            // pcl::transformPointCloud(*this->cloud, *this->cloud, tf);
            this->transformComputed = true;
        }

        void StampedPointCloud::applyIcpTransform(const Eigen::Matrix4f& tf) {

            if(!icpTransformComputed) {

                Eigen::Matrix4d mat4d = tf.cast<double>();
                Eigen::Affine3d affine(mat4d);

                this->applyTransform(affine);

                this->icpTransformComputed = true;
            }
        }

        void StampedPointCloud::removePointsWithLabel(std::uint32_t label) {

            std::lock_guard<std::mutex> lock(this->cloudMutex);

            auto it = this->cloud->begin();
            while (it != this->cloud->end()) {
                if (it->label == label)
                    it = this->cloud->erase(it);
                else
                    ++it;
            }
        }

        void StampedPointCloud::removePointsWithLabels(const std::set<std::uint32_t>& labels) {

            std::lock_guard<std::mutex> lock(this->cloudMutex);

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

            std::lock_guard<std::mutex> lock(this->cloudMutex);

            pcl::VoxelGrid<pcl::PointXYZRGBL> voxelGrid;
            voxelGrid.setInputCloud(this->cloud);
            voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
            voxelGrid.filter(*this->cloud);
        }
    } // pcl_aggregator
} // entities