
add_executable(benchmark
        benchmark.cpp
        testFullPipeLine.cpp
        testFullCNN.cpp
)

# 在构建目录输出文件
set_target_properties(benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(benchmark base neon ${OpenCV_LIBS})


# 设置include目录
target_include_directories(benchmark PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../base
        ${CMAKE_CURRENT_SOURCE_DIR}/../neon
)