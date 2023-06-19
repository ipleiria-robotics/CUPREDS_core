//
// Created by carlostojal on 30-04-2023.
//

#ifndef PCL_AGGREGATOR_CORE_UTILS_H
#define PCL_AGGREGATOR_CORE_UTILS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>

namespace pcl_aggregator {
    namespace utils {
        /*! \brief Utils.
         *         General utilities related primarily to timestamps and PointClouds.
         *
         */
        class Utils {

            public:
                /*! \brief Get current UNIX timestamp in milliseconds. */
                static unsigned long long getCurrentTimeMillis();

                /*! \brief Get age of a given timestamp in seconds. */
                static unsigned long long getAgeInSecs(unsigned long long timestamp);

                /*! \brief Get the UNIX timestamp of something with a given age.
                 *
                 * @param age Age in seconds.
                 * @return UNIX timestamp in milliseconds.
                 */
                static unsigned long long getMaxTimestampForAge(double age);

                /*! \brief Remove the points of a PointCloud from another. */
                static void removePointCloudFromOther(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud,
                                                      pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointsToRemove);
        };
    } // pcl_aggregator
} // utils


#endif //PCL_AGGREGATOR_CORE_UTILS_H
