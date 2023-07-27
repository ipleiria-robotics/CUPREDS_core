Student:<br>
Carlos Tojal<sup>1,2</sup><br>

Advisors:<br>
Luís Conde Bento<sup>1,2</sup><br>

<sup>1</sup> Polytechnic of Leiria, Leiria, Portugal;<br>
<sup>2</sup> Institute of Systems and Robotics, Coimbra, Portugal 

# CUPREDS – CUDA-Accelerated, Multi-threaded Point Cloud Registration from Heterogeneous Sensors with Efficient Data Structures

The is the core library of CUPREDS. There is a ROS wrapper developed, available [here](https://github.com/ipleiria-robotics/CUPREDS_ros).

This library implements all the core functionality, in case one wants to link it against another wrapper.

## Requirements

- C++20 compiler.
- CUDA Toolkit (75 architecture).
- Point Cloud Library (PCL).
- OpenCV
- Eigen 3
- Doxygen

## Compiling and installing

- In this directory:
    - `mkdir cmake-build-debug`
    - `cd cmake-build-debug`
- Run `cmake ..`. This will generate the Makefiles from the CMakeLists.
- Run `sudo make install`. This will build the library and install it in your system.
