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

#include <pcl_aggregator_core/cuda/CUDAPointClouds.h>

namespace pcl_aggregator {
    namespace cuda {
        namespace pointclouds {

            static __host__ int setPointCloudLabelCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, std::uint32_t label) {
                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                // declare the device input point array
                pcl::PointXYZRGBL *d_cloud;

                if((err = cudaSetDevice(0)) != cudaSuccess) {
                    std::cerr << "Error setting the CUDA device: " << cudaGetErrorString(err) << std::endl;
                    return -1;
                }

                // create a stream
                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating the label-setting CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return -2;
                }

                // allocate memory on the device to store the input pointcloud
                if ((err = cudaMalloc(&d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for the pointcloud: " << cudaGetErrorString(err) << std::endl;
                    return -3;
                }

                // copy the input pointcloud to the device
                if ((err = cudaMemcpy(d_cloud, cloud->points.data(), cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying the input pointcloud to the device (set label): " << cudaGetErrorString(err)
                              << std::endl;
                    return -4;
                }

                // call the kernel
                dim3 block(512);
                dim3 grid((cloud->size() + block.x - 1) / block.x);
                setPointLabelKernel<<<grid, block, 0, stream>>>(d_cloud, label, cloud->size());

                // wait for the stream
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the label-setting stream: " << cudaGetErrorString(err) << std::endl;
                    return -5;
                }

                // copy the output pointcloud back to the host
                if ((err = cudaMemcpy(cloud->points.data(), d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the output pointcloud to the host (labelling): " << cudaGetErrorString(err)
                              << std::endl;
                    return -6;
                }

                // free the memory
                if ((err = cudaFree(d_cloud)) != cudaSuccess) {
                    std::cerr << "Error freeing the pointcloud from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return -7;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return -8;
                }

                return 0;
            }

            static __global__ void setPointLabelKernel(pcl::PointXYZRGBL *points, std::uint32_t label, int num_points) {
                std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_points) {
                    points[idx].label = label;
                }
            }

            static __host__ int transformPointCloudCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, const Eigen::Affine3d& tf) {

                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                if((err = cudaSetDevice(0)) != cudaSuccess) {
                    std::cerr << "Error setting the CUDA device: " << cudaGetErrorString(err) << std::endl;
                    return -1;
                }

                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating pointcloud transform stream: " << cudaGetErrorString(err) << std::endl;
                    return -2;
                }

                // allocate device memory for the pointcloud
                pcl::PointXYZRGBL *d_cloud;
                if ((err = cudaMalloc(&d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for the pointcloud: " << cudaGetErrorString(err) << std::endl;
                    return -3;
                }

                // copy the pointcloud to the device
                if ((err = cudaMemcpy(d_cloud, cloud->points.data(), cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying the input pointcloud to the device (transform): " << cudaGetErrorString(err)
                              << std::endl;
                    return -4;
                }

                // call the kernel
                dim3 block(512);
                dim3 grid((cloud->size() + block.x - 1) / block.x);
                transformPointKernel<<<grid, block, 0, stream>>>(d_cloud, tf.matrix(), cloud->size());

                // wait for the stream
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the transform stream: " << cudaGetErrorString(err) << std::endl;
                    return -5;
                }

                // copy the output pointcloud back to the host
                if ((err = cudaMemcpy(cloud->points.data(), d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the output pointcloud to the host (transform): " << cudaGetErrorString(err)
                              << std::endl;
                    return -6;
                }

                // free the memory
                if ((err = cudaFree(d_cloud)) != cudaSuccess) {
                    std::cerr << "Error freeing the pointcloud from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return -7;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return -8;
                }

                return 0;
            }

            static __global__ void transformPointKernel(pcl::PointXYZRGBL *points, Eigen::Matrix4d transform, int num_points) {
                std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_points) {
                    Eigen::Vector4d p(points[idx].x, points[idx].y, points[idx].z, 1.0f);
                    p = transform * p;
                    points[idx].x = p(0);
                    points[idx].y = p(1);
                    points[idx].z = p(2);
                }
            }

            static __host__ int concatenatePointCloudsCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud1,
                                                     const pcl::PointCloud<pcl::PointXYZRGBL>& cloud2) {

                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                if((err = cudaSetDevice(0)) != cudaSuccess) {
                    std::cerr << "Error setting the CUDA device: " << cudaGetErrorString(err) << std::endl;
                    return -1;
                }

                // create a stream
                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating pointcloud concatenation stream: " << cudaGetErrorString(err) << std::endl;
                    return -2;
                }

                // resize the cloud1
                std::size_t cloud1OriginalSize = cloud1->size();
                std::size_t cloud1NewSize = cloud1OriginalSize + cloud2.size();
                cloud1->resize(cloud1NewSize);

                // allocate cloud1 on the device - allocate with sufficient space to the concatenation
                pcl::PointXYZRGBL *d_cloud1;
                if ((err = cudaMalloc(&d_cloud1, cloud1NewSize * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for cloud1: " << cudaGetErrorString(err) << std::endl;
                    return -3;
                }

                // copy cloud1 to the device
                if ((err = cudaMemcpy(d_cloud1, cloud1->points.data(), cloud1NewSize * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying cloud1 to the device: " << cudaGetErrorString(err)
                              << std::endl;
                    return -4;
                }

                // allocate cloud2 on the device
                pcl::PointXYZRGBL *d_cloud2;
                if((err = cudaMalloc(&d_cloud2, cloud2.size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for cloud2: " << cudaGetErrorString(err) << std::endl;
                    return -5;
                }

                // copy cloud2 to the device
                if((err = cudaMemcpy(d_cloud2, cloud2.points.data(), cloud2.size() * sizeof(pcl::PointXYZRGBL),
                                     cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying cloud2 to the device: " << cudaGetErrorString(err) << std::endl;
                    return -6;
                }

                // call the kernel
                dim3 block(512);
                // will be needed as much thread as the size of the cloud2, ideally
                dim3 grid((cloud2.size() + block.x - 1) / block.x);
                concatenatePointCloudsKernel<<<grid, block, 0, stream>>>(d_cloud1,
                                                                         cloud1OriginalSize, d_cloud2,
                                                                         cloud2.size());

                // wait for the stream to synchronize the threads
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the concatenation stream: " << cudaGetErrorString(err) << std::endl;
                    return -7;
                }

                // copy cloud1 back to the host
                if ((err = cudaMemcpy(cloud1->points.data(), d_cloud1, cloud1NewSize * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying cloud1 to the host: " << cudaGetErrorString(err)
                              << std::endl;
                    return -8;
                }

                // free cloud1
                if ((err = cudaFree(d_cloud1)) != cudaSuccess) {
                    std::cerr << "Error freeing cloud1 from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return -9;
                }

                // free cloud2
                if ((err = cudaFree(d_cloud2)) != cudaSuccess) {
                    std::cerr << "Error freeing cloud2 from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return -10;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return -11;
                }

                return 0;

            }

            static __global__ void concatenatePointCloudsKernel(pcl::PointXYZRGBL* cloud1, std::size_t cloud1_original_size,
                                                         pcl::PointXYZRGBL* cloud2, std::size_t cloud2_size) {
                // calculate the index
                std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                // check boundaries - should range between 0 and cloud2_size
                if(idx >= cloud2_size)
                    return;

                // copy the point from cloud2 to cloud1
                cloud1[cloud1_original_size+idx] = cloud2[idx];
            }
        }
    } // pcl_aggregator
} // cuda