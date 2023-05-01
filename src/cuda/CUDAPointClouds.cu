//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/cuda/CUDAPointClouds.cuh>

namespace pcl_aggregator {
    namespace cuda {
        namespace pointclouds {

            __host__ void setPointCloudLabelCuda(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, std::uint32_t label) {
                cudaError_t err = cudaSuccess;
                cudaStream_t stream;

                // declare the device input point array
                pcl::PointXYZRGBL *d_cloud;

                // create a stream
                if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating the CUDA stream: " << cudaGetErrorString(err) << std::endl;
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

            __host__ void transformPointCloudCuda(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, Eigen::Affine3d tf) {

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
        }
    } // pcl_aggregator
} // cuda