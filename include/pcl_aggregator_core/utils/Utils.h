//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_UTILS_H
#define PCL_AGGREGATOR_CORE_UTILS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_aggregator {
    namespace utils {
        class Utils {

            public:
                // get current unix timestamp in milliseconds
                static unsigned long long getCurrentTimeMillis();

                // get age to a timestamp in seconds
                static long getAgeInSecs(unsigned long long timestamp);

                // get the timestamp of a given age
                static unsigned long long getMaxTimestampForAge(double age);

                // remove the points of a pointcloud from another
                static void removePointCloudFromOther(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud,
                                                      pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointsToRemove);
        };
    } // pcl_aggregator
} // utils


#endif //PCL_AGGREGATOR_CORE_UTILS_H
