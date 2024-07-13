
set(SOURCES
        facedetectcnn.h
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
)
add_library(base STATIC ${SOURCES})
