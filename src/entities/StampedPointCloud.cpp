//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/entities/StampedPointCloud.h>
#include <pcl_aggregator_core/utils/Utils.h>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>

namespace pcl_aggregator {
    namespace entities {

        StampedPointCloud::StampedPointCloud(std::string originTopic) {
            this->timestamp = utils::Utils::getCurrentTimeMillis();

            this->setOriginTopic(originTopic);

            this->label = generateLabel();

            this->cloud = pcl::PointCloud<pcl::PointXYZRGBL>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBL>());
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

        typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr StampedPointCloud::getPointCloud() const {
            return this->cloud;
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
            this->cloud.reset();

            this->cloud = c;

            if(assignGeneratedLabel)
                this->assignLabelToPointCloud(this->cloud, this->label);
        }

        void StampedPointCloud::assignLabelToPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label) {

            cuda::pointclouds::setPointCloudLabelCuda(cloud, label);
        }

        void StampedPointCloud::setOriginTopic(const std::string& origin) {
            this->originTopic = origin;
        }

        bool StampedPointCloud::isTransformComputed() const {
            return this->transformComputed;
        }

        void StampedPointCloud::applyTransform(Eigen::Affine3d tf) {

            if(this->cloud == nullptr)
                return;

            cuda::pointclouds::transformPointCloudCuda(this->cloud, tf);

            // pcl::transformPointCloud(*this->cloud, *this->cloud, tf);
            this->transformComputed = true;
        }

        void StampedPointCloud::applyIcpTransform(Eigen::Matrix4f tf) {

            if(!icpTransformComputed) {

                Eigen::Matrix4d mat4d = tf.cast<double>();
                Eigen::Affine3d affine(mat4d);

                this->applyTransform(affine);

                this->icpTransformComputed = true;
            }
        }

        void StampedPointCloud::removePointsWithLabel(std::uint32_t label) {

            for(auto it = this->cloud->begin(); it != this->cloud->end(); it++) {
                if(it->label == label) {
                    this->cloud->erase(it);
                }
            }
        }
    } // pcl_aggregator
} // entities