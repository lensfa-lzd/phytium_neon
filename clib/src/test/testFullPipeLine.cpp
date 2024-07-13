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
#define DETECT_BUFFER_SIZE 0x20000


double benchmarkFullPipeLine(Mat image, int total_count,
                             short *benchFunction(unsigned char *result_buffer, unsigned char *rgb_image_data,
                                                  int width, int height,
                                                  int step)) {
    int num_thread = 1;
    short *pResults = NULL;
    unsigned char *pBuffers[1024];//large enough

    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char *p = (unsigned char *) malloc(DETECT_BUFFER_SIZE * num_thread);
    if (!p) {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }

    for (int i = 0; i < num_thread; i++) {
        pBuffers[i] = p + (DETECT_BUFFER_SIZE) * i;
    }

    // 确保数据加载到内存
    pResults = benchFunction(pBuffers[0], image.ptr<unsigned char>(0), (int) image.cols, (int) image.rows,
                             (int) image.step);

    TickMeter tm;
    tm.start();

    for (int i = 0; i < total_count; i++) {
        int idx = 0;
        pResults = benchFunction(pBuffers[idx], image.ptr<unsigned char>(0), (int) image.cols, (int) image.rows,
                                 (int) image.step);
    }
    tm.stop();
    double t = tm.getTimeMilli();
    t /= total_count;
    printf("Average time = %.2fms | %.2f FPS\n", t, 1000.0 / t);

    //release the buffer
    free(p);

    return t;
}

void testFullPipeLine(Mat image, int total_count) {
    printf("\n\n");
    printf("开始---------------全流程测试---------------开始\n");
    printf("基准       ");
    double t1 = benchmarkFullPipeLine(image, total_count, BASE::facedetect_cnn);
    printf("Neon加速   ");
    double t2 = benchmarkFullPipeLine(image, total_count, NeonACC::facedetect_cnn);
    printf("优化效率:   %.2f%%\n", 100 * (t1 - t2) / t1);
    printf("结束---------------全流程测试---------------结束\n");
    printf("\n\n");
}
