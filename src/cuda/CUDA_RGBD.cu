//
// Created by carlostojal on 30-04-2023.
//

#include <pcl_aggregator_core/cuda/CUDA_RGBD.cuh>

namespace pcl_aggregator {
    namespace cuda {
        namespace rgbd {

            __host__ void deprojectImages(const cv::Mat& colorImage, const cv::Mat& depthImage, const Eigen::Matrix3d& K, double minDepth, double maxDepth,
                                 pcl::PointCloud<pcl::PointXYZRGBL>::Ptr destination) {
                cudaError_t err;
                cudaStream_t stream;

                unsigned long long  nPixels = depthImage.total();
                unsigned long long nValidPoints = 0;
                pcl::PointXYZRGBL* pointArray;

                // declare device variable
                unsigned char *d_colorImage;
                unsigned char *d_depthImage;
                pcl::PointXYZRGBL *d_pointArray;

                unsigned long long *d_nValidPoints;

                // create the stream
                if((err = cudaStreamCreate(&stream)) != cudaSuccess) {
                    std::cerr << "Error creating the deprojection CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate the color image on device
                if((err = cudaMalloc(&d_colorImage, colorImage.total() * colorImage.elemSize())) != cudaSuccess) {
                    std::cerr << "Error allocating color image on device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate the depth image on device
                if((err = cudaMalloc(&d_depthImage, depthImage.total() * depthImage.elemSize())) != cudaSuccess) {
                    std::cerr << "Error allocating depth image on device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate the pointcloud on device
                if((err = cudaMalloc(&d_pointArray, nPixels * sizeof(pcl::PointXYZRGBL))) != cudaSuccess) {
                    std::cerr << "Error allocating pointcloud on device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate the valid point counter on device
                if((err = cudaMalloc(&d_nValidPoints, sizeof(size_t))) != cudaSuccess) {
                    std::cerr << "Error allocating number of valid points on device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the color image to the device
                if((err = cudaMemcpy(d_colorImage, colorImage.data, colorImage.total() * colorImage.elemSize(),
                                     cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying color image to device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the depth image to the device
                if((err = cudaMemcpy(d_depthImage, depthImage.data, depthImage.total() * depthImage.elemSize(),
                                     cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying depth image to device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the valid point counter to the device
                if((err = cudaMemcpy(d_nValidPoints, &nValidPoints, sizeof(unsigned long long), cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cerr << "Error copying the valid point counter to the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // call the kernel
                dim3 threadsPerBlock(16, 16); // 256 threads per block
                dim3 numBlocks(depthImage.rows / threadsPerBlock.x, depthImage.cols / threadsPerBlock.y);
                deprojectImagesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_colorImage, d_depthImage,
                                                                                 depthImage.cols, depthImage.rows,
                                                                                 K.inverse(), minDepth, maxDepth,
                                                                                 d_pointArray, d_nValidPoints);

                // wait for the stream to finish all threads
                if((err = cudaStreamSynchronize(stream)) != cudaSuccess) {
                    std::cerr << "Error synchronizing the deprojection CUDA stream: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the valid point counter back to the host
                if((err = cudaMemcpy(&nValidPoints, d_nValidPoints, sizeof(unsigned long long), cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the number of valid points back to the host: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // allocate a host point array
                if((pointArray = (pcl::PointXYZRGBL*) malloc(nValidPoints * sizeof(pcl::PointXYZRGBL))) == nullptr) {
                    std::cerr << "Error allocating host memory for the points: " << strerror(errno) << std::endl;
                    return;
                }

                // copy the point array back to the host
                if((err = cudaMemcpy(pointArray, d_pointArray, nPixels * sizeof(pcl::PointXYZRGBL), cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cerr << "Error copying the point array back to the host: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // copy the points to the host pointcloud
                for(size_t i = 0; i < nValidPoints; i++) {
                    destination->push_back(pointArray[i]);
                }
                destination.reset();

                // free the host point array
                free(pointArray);

                // free the color image from the device
                if((err = cudaFree(d_colorImage)) != cudaSuccess) {
                    std::cerr << "Error freeing the color image from the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // free the depth image from the device
                if((err = cudaFree(d_depthImage)) != cudaSuccess) {
                    std::cerr << "Error freeing the depth image from the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // free the point array from the device
                if((err = cudaFree(d_pointArray)) != cudaSuccess) {
                    std::cerr << "Error freeing the point array from the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                // free the valid point counter
                if((err = cudaFree(d_nValidPoints)) != cudaSuccess) {
                    std::cerr << "Error freeing the valid point counter from the device: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
            }

            __global__ void deprojectImagesKernel(unsigned char *colorImage, unsigned char *depthImage, unsigned int width,
                                                  unsigned int height, const Eigen::Matrix3d& K_inv,
                                                  double minDepth, double maxDepth, pcl::PointXYZRGBL *pointArray,
                                                  unsigned long long *nValidPoints) {

                // x: row
                // y: col

                int row = blockIdx.x * blockDim.x + threadIdx.x;
                int col = blockIdx.y * blockDim.y + threadIdx.y;

                if(row >= height || col >= width)
                    return;

                unsigned long long index = row * width + col;

                // declare the pixel as [x,y,d]
                Eigen::Vector3d pixel = {col, row, depthImage[index]};

                // deproject the point
                Eigen::Vector3d world_point = K_inv * pixel;

                // place the point on the last free spot
                pointArray[*nValidPoints].x = world_point.x();
                pointArray[*nValidPoints].y = world_point.y();
                pointArray[*nValidPoints].z = world_point.z();

                pointArray[*nValidPoints].b = colorImage[index * 3];
                pointArray[*nValidPoints].g = colorImage[index * 3 + 1];
                pointArray[*nValidPoints].r = colorImage[index * 3 + 2];

                // increment the valid point counter
                atomicAdd(nValidPoints, 1);
            }
        }
    } // pcl_aggregator
} // cuda