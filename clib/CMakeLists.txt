cmake_minimum_required(VERSION 3.0)


# 设置项目名称
project(FaceDetection)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加子目录
add_subdirectory(src)

# 添加可执行文件
#add_executable(benchmark benchmark.cpp)
#
## 将库链接到可执行文件
#target_link_libraries(benchmark test ${OpenCV_LIBS})
#
## 设置include目录
#target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

