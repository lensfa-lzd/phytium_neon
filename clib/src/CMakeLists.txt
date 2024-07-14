
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(base)
add_subdirectory(neon)
add_subdirectory(test)

# 定义动态库
add_library(faceDetect SHARED wrapper.cpp)

# 链接静态库到动态库
target_link_libraries(faceDetect
        base
        neon
)

# 设置库的输出路径
set_target_properties(faceDetect PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 动态库输出路径
        ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 静态库输出路径
)

# 设置include目录
target_include_directories(faceDetect PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/base
        ${CMAKE_CURRENT_SOURCE_DIR}/neon
)