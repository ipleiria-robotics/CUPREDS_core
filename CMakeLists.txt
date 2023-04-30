cmake_minimum_required(VERSION 3.25)
project(pcl_aggregator_core)

set(CMAKE_CXX_STANDARD 17)

add_library(pcl_aggregator_core SHARED library.cpp library.cpp library.h)
