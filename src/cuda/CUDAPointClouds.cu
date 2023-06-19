//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>

namespace pcl_aggregator {
    namespace cuda {
        namespace pointclouds {

            __host__ void setPointCloudLabelCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, std::uint32_t label) {
                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                // declare the device input point array
                pcl::PointXYZRGBL *d_cloud;

                // create a stream
                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating the label-setting CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate memory on the device to store the input pointcloud
                if ((err = cudaMalloc(&d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for the pointcloud: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the input pointcloud to the device
                if ((err = cudaMemcpy(d_cloud, cloud->points.data(), cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying the input pointcloud to the device: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // call the kernel
                dim3 block(512);
                dim3 grid((cloud->size() + block.x - 1) / block.x);
                setPointLabelKernel<<<grid, block, 0, stream>>>(d_cloud, label, cloud->size());

                // wait for the stream
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the output pointcloud back to the host
                if ((err = cudaMemcpy(cloud->points.data(), d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the output pointcloud to the host: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // free the memory
                if ((err = cudaFree(d_cloud)) != cudaSuccess) {
                    std::cerr << "Error freeing the pointcloud from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
            }

            __global__ void setPointLabelKernel(pcl::PointXYZRGBL *points, std::uint32_t label, int num_points) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_points) {
                    points[idx].label = label;
                }
            }

            __host__ void transformPointCloudCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud, const Eigen::Affine3d& tf) {

                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating pointcloud transform stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate device memory for the pointcloud
                pcl::PointXYZRGBL *d_cloud;
                if ((err = cudaMalloc(&d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for the pointcloud: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the pointcloud to the device
                if ((err = cudaMemcpy(d_cloud, cloud->points.data(), cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying the input pointcloud to the device: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // call the kernel
                dim3 block(512);
                dim3 grid((cloud->size() + block.x - 1) / block.x);
                transformPointKernel<<<grid, block, 0, stream>>>(d_cloud, tf.matrix(), cloud->size());

                // wait for the stream
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the output pointcloud back to the host
                if ((err = cudaMemcpy(cloud->points.data(), d_cloud, cloud->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the output pointcloud to the host: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // free the memory
                if ((err = cudaFree(d_cloud)) != cudaSuccess) {
                    std::cerr << "Error freeing the pointcloud from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
            }

            __global__ void transformPointKernel(pcl::PointXYZRGBL *points, Eigen::Matrix4d transform, int num_points) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_points) {
                    Eigen::Vector4d p(points[idx].x, points[idx].y, points[idx].z, 1.0f);
                    p = transform * p;
                    points[idx].x = p(0);
                    points[idx].y = p(1);
                    points[idx].z = p(2);
                }
            }

            __host__ void concatenatePointCloudsCuda(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud1,
                                                     const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& cloud2) {

                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                // create a stream
                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating pointcloud transform stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // resize the cloud1
                std::size_t cloud1OriginalSize = cloud1->size();
                cloud1->resize(cloud1OriginalSize + cloud2->size());

                // allocate cloud1 on the device - allocate with sufficient space to the concatenation
                pcl::PointXYZRGBL *d_cloud1;
                if ((err = cudaMalloc(&d_cloud1, cloud1->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for cloud1: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy cloud1 to the device
                if ((err = cudaMemcpy(d_cloud1, cloud1->points.data(), cloud1->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying cloud1 to the device: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // allocate cloud2 on the device
                pcl::PointXYZRGBL *d_cloud2;
                if((err = cudaMalloc(&d_cloud2, cloud2->size() * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating memory for cloud2: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy cloud2 to the device
                if((err = cudaMemcpy(d_cloud2, cloud2->points.data(), cloud2->size() * sizeof(pcl::PointXYZRGBL),
                                     cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying cloud2 to the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // call the kernel
                dim3 block(512);
                // will be needed as much thread as the size of the cloud2, ideally
                dim3 grid((cloud2->size() + block.x - 1) / block.x);
                concatenatePointCloudsKernel<<<grid, block, 0, stream>>>(d_cloud1,
                                                                         cloud1OriginalSize, d_cloud2,
                                                                         cloud2->size());

                // wait for the stream to synchronize the threads
                if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error waiting for the stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy cloud1 back to the host
                if ((err = cudaMemcpy(cloud1->points.data(), d_cloud1, cloud1->size() * sizeof(pcl::PointXYZRGBL),
                                      cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying cloud1 to the host: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // free cloud1
                if ((err = cudaFree(d_cloud1)) != cudaSuccess) {
                    std::cerr << "Error freeing cloud1 from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // free cloud2
                if ((err = cudaFree(d_cloud2)) != cudaSuccess) {
                    std::cerr << "Error freeing cloud2 from device memory: " << cudaGetErrorString(err)
                              << std::endl;
                    return;
                }

                // destroy the stream
                if ((err = cudaStreamDestroy(stream)) != cudaSuccess) {
                    std::cerr << "Error destroying the CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

            }

            __global__ void concatenatePointCloudsKernel(pcl::PointXYZRGBL* cloud1, std::size_t cloud1_original_size,
                                                         pcl::PointXYZRGBL* cloud2, std::size_t cloud2_size) {
                // calculate the index
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                // check boundaries - should range between 0 and cloud2_size
                if(idx >= cloud2_size)
                    return;

                // copy the point from cloud2 to cloud1
                cloud1[cloud1_original_size+idx] = cloud2[idx];
            }
        }
    } // pcl_aggregator
} // cuda