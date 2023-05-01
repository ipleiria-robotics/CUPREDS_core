cmake_minimum_required(VERSION 3.25)
project(pcl_aggregator_core)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED) # to process pointclouds
find_package(OpenCV REQUIRED) # to process image
find_package(Eigen3 REQUIRED) # to do linear algebra, i.e. transformations and linear equations
find_package(CUDA REQUIRED) # to do parallel programming
find_package(Doxygen REQUIRED) # to generate pretty and easy to read documentation

include_directories(include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_library(pcl_aggregator_core SHARED src/utils/Utils.cpp src/entities/StampedPointCloud.cpp src/managers/PointCloudsManager.cpp src/managers/StreamManager.cpp src/PCLRegistrator.cpp src/utils/RGBDDeprojector.cpp src/cuda/CUDAPointClouds.cu)

target_link_libraries(pcl_aggregator_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES} ${CUDA_LIBRARIES})

doxygen_add_docs(docs ${PROJECT_SOURCE_DIR})