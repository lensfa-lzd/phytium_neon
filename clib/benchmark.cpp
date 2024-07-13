#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
//#include "facedetectcnn.h"
#include "facedetectcnn_neon.h"

using namespace cv;
using namespace std;

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000


void benchmark(Mat image, int total_count,
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

int main(int argc, char *argv[]) {
    // argc: 表示参数个数，包括程序名本身(argv[0])。 在这个例子中， argc 的值是 4。
    // argv: 是一个 char* 类型的数组，存储了各个参数字符串。
    if (argc == 1 || argc > 3) {
        printf("Usage: %s <image_file_name>\n", argv[0]);
        printf("Usage: %s <image_file_name> <repeat_count>\n", argv[0]);
        return -1;
    }
    // 重复的图像处理次数
    int total_count;
    if (argc == 2) {
        total_count = 4;
    } else {
        total_count = stoi(argv[2]);
    }
    printf("Repeat %d times.\n", total_count);

    //load an image and convert it to gray (single-channel)
    Mat image = imread(argv[1]);
    if (image.empty()) {
        fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
        return -1;
    }

    int num_thread = 1;
    printf("There is %d thread.\n", num_thread);

    printf("Benchmarking...\n");
    printf("facedetect_cnn\n");
//    benchmark(image, total_count, facedetect_cnn);
    printf("----\n");
    printf("facedetect_cnn_neon\n");
    benchmark(image, total_count, NeonACC::facedetect_cnn_neon);

    return 0;
}

