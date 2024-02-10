cmake_minimum_required(VERSION 3.16)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(pcl_aggregator_core LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(USE_CUDA ON)
set(USE_ICP ON)

if(USE_ICP)
    add_compile_definitions(USE_ICP)
endif()

find_package(PCL REQUIRED) # to process pointclouds
find_package(OpenCV REQUIRED) # to process image
find_package(Eigen3 REQUIRED) # to do linear algebra, i.e. transformations and linear equations
if(USE_CUDA)
    find_package(CUDA REQUIRED) # to do parallel programming
endif()

set(SOURCE_FILES src/utils/Utils.cpp src/entities/StampedPointCloud.cpp
        src/utils/RGBDDeprojector.cpp src/managers/IntraSensorManager.cpp src/managers/InterSensorManager.cpp)
if(USE_CUDA)
    message(STATUS "Compiling with CUDA support")
    set(CUDA_SOURCE_FILES src/cuda/CUDA_RGBD.cu src/cuda/CUDAPointClouds.cu)
    set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    add_compile_definitions(USE_CUDA)
    # append cuda source files
    set(SOURCE_FILES ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
endif()

set(PUBLIC_HEADERS include/pcl_aggregator_core)
include_directories(include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_library(pcl_aggregator_core SHARED ${SOURCE_FILES})

set(LINK_LIBRARIES ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES})
if(USE_CUDA)
    set(LINK_LIBRARIES "${LINK_LIBRARIES} ${CUDA_LIBRARIES}")
endif()
target_link_libraries(pcl_aggregator_core ${LINK_LIBRARIES})

install(TARGETS pcl_aggregator_core LIBRARY DESTINATION /usr/lib/pcl_aggregator_core)
install(DIRECTORY ${PUBLIC_HEADERS} DESTINATION /usr/include)
install(FILES pcl_aggregator_coreConfig.cmake DESTINATION /usr/lib/pcl_aggregator_core)
