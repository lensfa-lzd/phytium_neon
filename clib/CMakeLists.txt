cmake_minimum_required(VERSION 3.0)


# 设置项目名称
project(FaceDetection)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(benchmark benchmark.cpp)

# 添加源文件
add_library(facedetectcnn_v1
        src/base/facedetectcnn.cpp
        src/base/facedetectcnn-data.cpp
        src/base/facedetectcnn-model.cpp)

add_library(facedetectcnn_neon_v2
        src/neon/facedetectcnn_neon.cpp
        src/neon/facedetectcnn-data_neon.cpp
        src/neon/facedetectcnn-model_neon.cpp)

# 将库链接到可执行文件
target_link_libraries(benchmark facedetectcnn facedetectcnn_neon ${OpenCV_LIBS})

target_include_directories(facedetectcnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/base)
target_include_directories(facedetectcnn_neon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/neon)

