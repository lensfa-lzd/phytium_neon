
set(SOURCES
        facedetectcnn_neon.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(neon STATIC ${SOURCES})
target_compile_options(neon PRIVATE -fPIC)

# 设置库的输出路径
set_target_properties(neon PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 动态库输出路径
        ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 静态库输出路径
)