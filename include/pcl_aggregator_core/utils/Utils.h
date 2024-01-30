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
