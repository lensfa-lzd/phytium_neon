cmake_minimum_required(VERSION 3.0)

set(ENABLE_NEON ON)
message("NEON = ${ENABLE_NEON}")

if(ENABLE_NEON)
    add_definitions(-D_ENABLE_NEON)
endif()

# 设置项目名称
project(FaceDetection)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(benchmark benchmark.cpp)

# 添加源文件
add_library(facedetectcnn
        facedetectcnn.cpp
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp)

# 将库链接到可执行文件
target_link_libraries(benchmark facedetectcnn ${OpenCV_LIBS})
