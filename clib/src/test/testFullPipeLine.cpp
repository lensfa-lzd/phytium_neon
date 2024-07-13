//
// Created by liang on 2024/7/13.
//

#include "iostream"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"

using namespace cv;
using namespace std;

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000


void benchmarkFullPipeLine(Mat image, int total_count,
               int *benchFunction(unsigned char *result_buffer, unsigned char *rgb_image_data, int width, int height,
                                  int step)) {
    int num_thread = 1;
    int *pResults = NULL;
    unsigned char *pBuffers[1024];//large enough

    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char *p = (unsigned char *) malloc(DETECT_BUFFER_SIZE * num_thread);
    if (!p) {
        fprintf(stderr, "Can not alloc buffer.\n");
        return;
    }

    for (int i = 0; i < num_thread; i++) {
        pBuffers[i] = p + (DETECT_BUFFER_SIZE) * i;
    }


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
    printf("cnn facedetection average time = %.2fms | %.2f FPS\n", t, 1000.0 / t);

    //release the buffer
    free(p);
}

void testFullPipeLine(Mat image, int total_count) {
    int num_thread = 1;
    printf("Using %d thread.\n", num_thread);

    printf("Benchmarking...\n");
    printf("facedetect_cnn\n");
    benchmarkFullPipeLine(image, total_count, BASE::facedetect_cnn);
    printf("----\n");
    printf("facedetect_cnn_neon\n");
    benchmarkFullPipeLine(image, total_count, NeonACC::facedetect_cnn);
}
