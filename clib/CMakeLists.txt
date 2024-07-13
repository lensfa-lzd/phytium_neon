cmake_minimum_required(VERSION 3.0)


# 设置项目名称
project(FaceDetection)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)


# 添加子目录
add_subdirectory(src)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detectImage detect-image.cpp)
target_link_libraries(detectImage base ${OpenCV_LIBS})

target_include_directories(detectImage PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/base
)