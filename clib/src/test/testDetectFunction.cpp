//
// Created by liang on 2024/7/13.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"

using namespace cv;
using namespace std;


double benchmarkDetectFunction(unsigned char *rgb_image_data, int width, int height, int step, int total_count, int function_type,
                               void (*benchFunction)(unsigned char *rgbImageData, int width,
                                                     int height, int step, int function_type)) {
    // 确保数据加载到内存
    benchFunction(rgb_image_data, width, height, step, function_type);

    TickMeter tm;
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

void functionContainer(unsigned char *rgb_image_data, int width, int height, int step, int function_type) {
    if (function_type == 0) {
        BASE::objectdetect_cnn(rgb_image_data, width, height, step);
    } else {
        NeonACC::objectdetect_cnn(rgb_image_data, width, height, step);
    }
}

void testDetectFunction(unsigned char *rgb_image_data, int width, int height, int step, int total_count) {
    printf("\n");
    printf("开始---------------检测函数测试---------------开始\n");
    printf("基准       ");
    double t1 = benchmarkDetectFunction(rgb_image_data, width, height, step, total_count, 0, functionContainer);
    printf("Neon加速   ");
    double t2 = benchmarkDetectFunction(rgb_image_data, width, height, step, total_count, 1, functionContainer);
    printf("优化效率:   %.2f%%\n", 100 * (t1 - t2) / t1);
    printf("结束---------------检测函数测试---------------结束\n");
    printf("\n");
}
