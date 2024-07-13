#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "test.h"

using namespace cv;
using namespace std;


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
    printf("Using image %s\n", argv[1]);
    printf("Repeat %d times.\n", total_count);

    //load an image and convert it to gray (single-channel)
    Mat image = imread(argv[1]);
    if (image.empty()) {
        fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
        return -1;
    }
    testFullPipeLine(image, total_count);

    return 0;
}

