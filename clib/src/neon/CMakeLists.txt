
set(SOURCES
        facedetectcnn_neon.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(neon STATIC ${SOURCES})
