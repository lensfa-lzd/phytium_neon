//
// Created by liang on 2024/7/13.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"
#include <fstream>


double benchmarkDetectFunction(unsigned char *rgb_image_data, int width, int height, int step, int total_count,
                               int function_type,
                               void (*benchFunction)(unsigned char *rgbImageData, int width,
                                                     int height, int step, int function_type)) {
    // 确保数据加载到内存
    benchFunction(rgb_image_data, width, height, step, function_type);

    cv::TickMeter tm;
    tm.start();

    for (int i = 0; i < total_count; i++) {
        benchFunction(rgb_image_data, width, height, step, function_type);
    }
    tm.stop();
    double t = tm.getTimeMilli();
    t /= total_count;
    printf("Average time = %.2fms | %.2f FPS\n", t, 1000.0 / t);
    return t;
}

void functionContainerDetect(unsigned char *rgb_image_data, int width, int height, int step, int function_type) {
    if (function_type == 0) {
        BASE::objectdetect_cnn(rgb_image_data, width, height, step);
    } else {
        NeonACC::objectdetect_cnn(rgb_image_data, width, height, step);
    }
}

cv::Mat parmGenerator(int mode) {
    // 创建一个空图像，每个像素包含 3 个通道 (BGR)，每个通道 8 位无符号整数
    cv::Mat image;
    if (mode == 0) {
        image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
    } else if (mode == 1) {
        image = cv::Mat::zeros(cv::Size(320, 240), CV_8UC3);
    } else if (mode == 2) {
        image = cv::Mat::zeros(cv::Size(160, 120), CV_8UC3);
    } else {
        image = cv::Mat::zeros(cv::Size(80, 60), CV_8UC3);
    }

    // 生成 0-255 之间的随机值并填充图像
    cv::randu(image, cv::Scalar(0), cv::Scalar(255));
    return image;
}

double testDetectFunction_(unsigned char *rgb_image_data, int width, int height, int step, int total_count) {
    printf("测试图像大小: (%d, %d)\n", width, height);
    printf("基准       ");
    double t1 = benchmarkDetectFunction(rgb_image_data, width, height, step, total_count, 0, functionContainerDetect);
    printf("Neon加速   ");
    double t2 = benchmarkDetectFunction(rgb_image_data, width, height, step, total_count, 1, functionContainerDetect);

    double efficiency = 100 * (t1 - t2) / t1;
    printf("优化效率:   %.2f%%\n\n", efficiency);
    return efficiency;
}


void testDetectFunction(int total_count, std::ofstream& resultFile) {
    // 测试多次不同大小的图像
    printf("\n");
    printf("开始---------------检测函数测试---------------开始\n");

    double efficiency[4] = {0};
    for (int i = 0; i < 4; i++) {
        cv::Mat image = parmGenerator(i);
        efficiency[i] = testDetectFunction_(image.data, image.cols, image.rows, (int) image.step, total_count);
    }


    printf("结束---------------检测函数测试---------------结束\n");
    printf("\n");

    std::string image_size[4] = {"640x480", "320x240", "160x120", "80x60"};
    if (resultFile.is_open()) {
        resultFile << "DetectFunction:" << std::endl;
        for (int i = 0; i < 4; i++) {
            resultFile << "  " << image_size[i] << ": " << efficiency[i] << std::endl;
        }
    }
}