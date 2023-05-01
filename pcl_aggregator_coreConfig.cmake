# Check if this file has already been included.
if(NOT TARGET pcl_aggregator_core)
    # Define the target.
    add_library(pcl_aggregator_core SHARED IMPORTED)
    set_target_properties(pcl_aggregator_core PROPERTIES IMPORTED_LOCATION /usr/lib/pcl_aggregator_core/libpcl_aggregator_core.so)

    # Set the dependencies.
    find_package(PCL REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(CUDA REQUIRED)

    target_include_directories(pcl_aggregator_core INTERFACE /usr/include/pcl_aggregator_core)
    target_link_libraries(pcl_aggregator_core INTERFACE ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES} ${CUDA_LIBRARIES})
endif()