//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/entities/StampedPointCloud.h>
#include <pcl_aggregator_core/utils/Utils.h>
#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>
#include <utility>

namespace pcl_aggregator {
    namespace entities {

        template <typename PointTypeT>
        StampedPointCloud<PointTypeT>::StampedPointCloud(std::string originTopic) {
            this->timestamp = utils::Utils::getCurrentTimeMillis();

            this->setOriginTopic(originTopic);

            this->label = generateLabel();

            this->cloud = pcl::PointCloud<PointTypeT>::Ptr(new pcl::PointCloud<PointTypeT>());
        }

        // generate a 32-bit label and assign
        template <typename PointTypeT>
        std::uint32_t StampedPointCloud<PointTypeT>::generateLabel() {

            std::string combined = this->originTopic + std::to_string(this->timestamp);

            std::hash<std::string> hasher;
            std::uint32_t hash_value = hasher(combined);

            return hash_value;
        }

        template <typename PointTypeT>
        unsigned long long StampedPointCloud<PointTypeT>::getTimestamp() const {
            return this->timestamp;
        }

        template <typename PointTypeT>
        typename pcl::PointCloud<PointTypeT>::Ptr StampedPointCloud<PointTypeT>::getPointCloud() const {
            return this->cloud;
        }

        template <typename PointTypeT>
        std::string StampedPointCloud<PointTypeT>::getOriginTopic() const {
            return this->originTopic;
        }

        template <typename PointTypeT>
        std::uint32_t StampedPointCloud<PointTypeT>::getLabel() const {
            return this->label;
        }

        template <typename PointTypeT>
        bool StampedPointCloud<PointTypeT>::isIcpTransformComputed() const {
            return icpTransformComputed;
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::setTimestamp(unsigned long long t) {
            this->timestamp = t;
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::setPointCloud(typename pcl::PointCloud<PointTypeT>::Ptr c, bool assignGeneratedLabel) {
            this->cloud.reset();

            this->cloudSet = true;
            this->cloud = c;

            if(assignGeneratedLabel)
                this->assignLabelToPointCloud(this->cloud, this->label);
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::assignLabelToPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label) {

            cuda::pointclouds::setPointCloudLabelCuda(cloud, label);
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::setOriginTopic(const std::string& origin) {
            this->originTopic = origin;
        }

        template <typename PointTypeT>
        bool StampedPointCloud<PointTypeT>::isTransformComputed() const {
            return this->transformComputed;
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::applyTransform(Eigen::Affine3d tf) {
            if(this->cloudSet) {

                transformPointCloudCuda(this->cloud, tf);

                // pcl::transformPointCloud(*this->cloud, *this->cloud, tf);
                this->transformComputed = true;
            }
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::applyIcpTransform(Eigen::Matrix4f tf) {

            if(!icpTransformComputed) {

                Eigen::Matrix4d mat4d = tf.cast<double>();
                Eigen::Affine3d affine(mat4d);

                this->applyTransform(affine);

                this->icpTransformComputed = true;
            }
        }

        template <typename PointTypeT>
        void StampedPointCloud<PointTypeT>::removePointsWithLabel(std::uint32_t label) {

            for(auto it = this->cloud->begin(); it != this->cloud->end(); it++) {
                if(it->label == label) {
                    this->cloud->erase(it);
                }
            }
        }
    } // pcl_aggregator
} // entities