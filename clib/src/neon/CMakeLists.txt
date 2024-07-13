
set(SOURCES
        facedetectcnn_neon.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(neon SHARED ${SOURCES})

# 设置库的输出路径
set_target_properties(neon PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)