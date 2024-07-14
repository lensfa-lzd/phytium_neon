//
// Created by liang on 2024/7/14.
//

#include <iostream>
#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"

extern "C" void hello() {
    std::cout << "faceDetectLib loaded!" << std::endl;
}

extern "C" short *faceDetectBase(
        unsigned char *result_buffer,
        unsigned char *rgb_image_data, int width, int height, int step){
    return BASE::facedetect_cnn(result_buffer, rgb_image_data, width, height, step);
}

extern "C" short *faceDetectNeon(
        unsigned char *result_buffer,
        unsigned char *rgb_image_data, int width, int height, int step){
    return NeonACC::facedetect_cnn(result_buffer, rgb_image_data, width, height, step);
}