//
// Created by liang on 2024/7/13.
//

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"

using namespace cv;
using namespace std;

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x800


double benchmarkFullPipeLine(Mat image, int total_count,
                             short *(*benchFunction)(unsigned char *result_buffer, unsigned char *rgb_image_data,
                                                  int width, int height,
                                                  int step)) {

    unsigned char *pBuffer = (unsigned char *) malloc(DETECT_BUFFER_SIZE);


    if (!pBuffer) {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }


    // 确保数据加载到内存
    //    benchFunction(pBuffer, image.ptr<unsigned char>(0), (int) image.cols, (int) image.rows,
    //                             (int) image.step);

    TickMeter tm;
    tm.start();

    for (int i = 0; i < total_count; i++) {
        benchFunction(pBuffer, image.ptr<unsigned char>(0), (int) image.cols, (int) image.rows,
                                 (int) image.step);
    }
    tm.stop();
    double t = tm.getTimeMilli();
    t /= total_count;
    printf("Average time = %.2fms | %.2f FPS\n", t, 1000.0 / t);

    //release the buffer
    free(pBuffer);

    return t;
}

void testFullPipeLine(Mat image, int total_count) {
    printf("\n");
    printf("开始---------------全流程测试---------------开始\n");
    printf("基准       ");
    double t1 = benchmarkFullPipeLine(image, total_count, BASE::facedetect_cnn);
    printf("Neon加速   ");
    double t2 = benchmarkFullPipeLine(image, total_count, NeonACC::facedetect_cnn_neon);
    printf("优化效率:   %.2f%%\n", 100 * (t1 - t2) / t1);
    printf("结束---------------全流程测试---------------结束\n");
    printf("\n");
}
