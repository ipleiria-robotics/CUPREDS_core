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

        typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr StampedPointCloud::getPointCloud() {
            this->cloudMutex.lock();
            pcl::PointCloud<pcl::PointXYZRGBL>::Ptr c = this->cloud;
            this->cloudMutex.unlock();
            return c;
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

        void StampedPointCloud::setPointCloud(const typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& c, bool assignGeneratedLabel) {
            this->cloudMutex.lock();

            // free the old pointcloud
            this->cloud.reset();

            // set the new
            this->cloud = c;

            if(assignGeneratedLabel)
                StampedPointCloud::assignLabelToPointCloud(this->cloud, this->label);

            this->cloudMutex.unlock();
        }

        void StampedPointCloud::assignLabelToPointCloud(typename pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label) {

            cuda::pointclouds::setPointCloudLabelCuda(std::move(cloud), label);
        }

        void StampedPointCloud::setOriginTopic(const std::string& origin) {
            this->originTopic = origin;
        }

        bool StampedPointCloud::isTransformComputed() const {
            return this->transformComputed;
        }

        void StampedPointCloud::applyTransform(Eigen::Affine3d tf) {

            this->cloudMutex.lock();

            if(this->cloud == nullptr)
                return;

            cuda::pointclouds::transformPointCloudCuda(this->cloud, tf);

            this->cloudMutex.unlock();

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

            this->cloudMutex.lock();

            auto it = this->cloud->begin();
            while (it != this->cloud->end()) {
                if (it->label == label)
                    it = this->cloud->erase(it);
                else
                    ++it;
            }

            this->cloudMutex.unlock();
        }
    } // pcl_aggregator
} // entities