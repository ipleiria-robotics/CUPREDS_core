//
// Created by carlostojal on 01-05-2023.
//

#include <pcl_aggregator_core/managers/PointCloudsManager.h>

namespace pcl_aggregator {
    namespace managers {

        PointCloudsManager::PointCloudsManager(size_t nSources, double maxAge) {
            this->nSources = nSources;

            // initialize empty merged cloud
            this->mergedCloud = pcl::PointCloud<PointTypeT>::Ptr(new pcl::PointCloud<PointTypeT>());

            this->maxAge = maxAge;
        }

        PointCloudsManager::~PointCloudsManager() {
            // free the merged pointcloud
            this->mergedCloud.reset();

            // free all the stream managers
            for(auto iter = this->streamManagers.begin(); iter != this->streamManagers.end(); ++iter) {
                iter->second.reset();
            }
        }

        size_t PointCloudsManager::getNClouds() const {
            return this->nSources;
        }

        void PointCloudsManager::addCloud(const pcl::PointCloud<PointTypeT>::Ptr &cloud, const std::string &topicName) {

            // check if the pointcloud is null or empty
            if(cloud == nullptr)
                return;
            if(cloud->empty())
                return;

            // the key is not present
            this->initStreamManager(topicName, this->maxAge);

            this->streamManagers[topicName]->addCloud(cloud);

        }

        void PointCloudsManager::setTransform(const Eigen::Affine3d &transform, const std::string &topicName) {
            this->initStreamManager(topicName, this->maxAge);

            this->streamManagers[topicName]->setSensorTransform(transform);
        }

        pcl::PointCloud<PointTypeT> PointCloudsManager::getMergedCloud() {
            // clear the old merged cloud
            this->clearMergedCloud();

            bool firstCloud = true;

            // iterate the map
            /* TODO: review performance of only perform merging on demand
             * vs merging the pointclouds and removing as needed every time
            */
            std::lock_guard<std::mutex> lock(this->managersMutex);
            for(auto iter = this->streamManagers.begin(); iter != this->streamManagers.end(); ++iter) {
                if(firstCloud) {
                    this->mergedCloud = iter->second->getCloud();
                    firstCloud = false;
                } else {
                    this->appendToMerged(iter->second->getCloud());
                }
            }

            // this->downsampleMergedCloud();
            return *this->mergedCloud;
        }

        bool PointCloudsManager::appendToMerged(const pcl::PointCloud<PointTypeT>::Ptr &input) {
            return false;
        }


    } // pcl_aggregator
} // managers