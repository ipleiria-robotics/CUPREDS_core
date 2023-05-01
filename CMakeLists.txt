cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(pcl_aggregator_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(PCL REQUIRED) # to process pointclouds
find_package(OpenCV REQUIRED) # to process image
find_package(Eigen3 REQUIRED) # to do linear algebra, i.e. transformations and linear equations
find_package(CUDA REQUIRED) # to do parallel programming
find_package(Doxygen REQUIRED) # to generate pretty and easy to read documentation

set(PUBLIC_HEADERS include/pcl_aggregator_core)
include_directories(include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_library(pcl_aggregator_core SHARED src/utils/Utils.cpp src/entities/StampedPointCloud.cpp src/utils/RGBDDeprojector.cpp src/cuda/CUDAPointClouds.cu src/managers/StreamManager.cpp src/managers/PointCloudsManager.cpp)

target_link_libraries(pcl_aggregator_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES} ${CUDA_LIBRARIES})

doxygen_add_docs(docs ${PROJECT_SOURCE_DIR})

install(TARGETS pcl_aggregator_core LIBRARY DESTINATION /usr/lib/pcl_aggregator_core)
install(DIRECTORY ${PUBLIC_HEADERS} DESTINATION /usr/include)