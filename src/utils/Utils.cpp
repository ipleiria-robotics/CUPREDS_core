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

#include <pcl_aggregator_core/utils/Utils.h>

namespace pcl_aggregator {
    namespace utils {

        unsigned long long Utils::getCurrentTimeMillis() {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        }

        unsigned long long Utils::getAgeInSecs(unsigned long long timestamp) {
            return (Utils::getCurrentTimeMillis() - timestamp) / 1000;
        }

        unsigned long long Utils::getMaxTimestampForAge(double age) {
            return Utils::getCurrentTimeMillis() - (unsigned long long) (age * 1000);
        }

        void Utils::removePointCloudFromOther(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud,
                                              pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointsToRemove) {
            // TODO
        }
    } // pcl_aggregator
} // entities