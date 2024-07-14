
set(SOURCES
        facedetectcnn.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(base STATIC ${SOURCES})
target_compile_options(base PRIVATE -fPIC)

# 设置库的输出路径
set_target_properties(base PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 动态库输出路径
        ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"  # 静态库输出路径
)