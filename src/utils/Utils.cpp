//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/utils/Utils.h>

namespace pcl_aggregator {
    namespace utils {

        unsigned long long Utils::getCurrentTimeMillis() {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        }

        long Utils::getAgeInSecs(unsigned long long timestamp) {
            return (Utils::getCurrentTimeMillis() - timestamp) / 1000;
        }

        unsigned long long Utils::getMaxTimestampForAge(double age) {
            return Utils::getCurrentTimeMillis() - (age * 1000);
        }

        void Utils::removePointCloudFromOther(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud,
                                              pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointsToRemove) {
            // TODO
        }
    } // pcl_aggregator
} // entities