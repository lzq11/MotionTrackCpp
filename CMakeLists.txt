cmake_minimum_required(VERSION 3.10)
project(motiontrack)
add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

# eigen
# 3060Lap
include_directories(/home/lzq/Lib/eigen-3.3.8)
# Jetson
include_directories(/home/nvidia/eigen-3.3.8)


# cuda
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
# 3060Lap
include_directories(/home/lzq/Lib/TensorRT-8.4.1.5/include)
link_directories(/home/lzq/Lib/TensorRT-8.4.1.5/lib)
# Jetson
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu//)

# opencv
find_package(OpenCV) 
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")#release
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")#debug

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(motiontrack ${My_Source_Files})
target_link_libraries(motiontrack nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(motiontrack cudart)
target_link_libraries(motiontrack ${OpenCV_LIBS})

# add_definitions(-O2 -pthread)