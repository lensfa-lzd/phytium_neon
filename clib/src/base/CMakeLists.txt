
set(SOURCES
        facedetectcnn.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(base SHARED ${SOURCES})

# 设置库的输出路径
set_target_properties(base PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)