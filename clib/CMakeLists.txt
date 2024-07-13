cmake_minimum_required(VERSION 3.0)


# 设置项目名称
project(FaceDetection)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/base)
set(NEON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/neon)

# 定义两个库的源文件
set(BASE_SRCS
        ${BASE_DIR}/facedetectcnn.cpp
        ${BASE_DIR}/facedetectcnn-data.cpp
        ${BASE_DIR}/facedetectcnn-model.cpp)

set(NEON_SRCS
        ${NEON_DIR}/facedetectcnn.cpp
        ${NEON_DIR}/facedetectcnn-data.cpp
        ${NEON_DIR}/facedetectcnn-model.cpp)

# 创建两个静态库
add_library(base STATIC ${BASE_SRCS})
add_library(neon STATIC ${NEON_SRCS})

# 添加可执行文件
add_executable(benchmark benchmark.cpp)

# 将库链接到可执行文件
target_link_libraries(benchmark base neon ${OpenCV_LIBS})

# 设置include目录
target_include_directories(benchmark PUBLIC ${BASE_DIR} ${NEON_DIR})

