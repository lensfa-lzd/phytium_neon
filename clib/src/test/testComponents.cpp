//
// Created by liang on 2024/7/14.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "arm_neon.h"
#include <random>
#include <fstream>

#define TEST_ARRAY_SIZE 1024
float *createRandomFloatArray(int size) {
    // 创建随机数引擎
    std::random_device rd; // 用于获取随机种子
    std::mt19937 gen(rd()); // Mersenne Twister 引擎

    // 创建随机数分布
    std::uniform_real_distribution<float> distrib(-1, 1);

    // 生成随机数组
    auto *arr = new float[size];
    for (int i = 0; i < size; i++) {
        arr[i] = distrib(gen);
    }
    return arr;
}

double benchmarkComponents(int total_count, int function_type,
                           void (*benchFunction)(float *p1, float *p2, float *p3, int num, int function_type)) {
    int test_size = TEST_ARRAY_SIZE;
    double total_time = 0;
    for (int i = 0; i < total_count; i++) {
        float *p1 = createRandomFloatArray(test_size);
        float *p2 = createRandomFloatArray(test_size);
        float *p3 = createRandomFloatArray(test_size);

        cv::TickMeter tm;
        tm.start();

        benchFunction(p1, p2, p3, test_size, function_type);

        tm.stop();
        double t = tm.getTimeMilli();
        total_time += t;

        delete[]p1;
        delete[]p2;
        delete[]p3;
    }
    total_time /= total_count;
    printf("Average time = %.4fms\n", total_time);
    return total_time;
}


void testDotProduct(float *p1, float *p2, float *p3, int num, int function_type) {
    if (function_type == 0) {
        float sum = 0.f;
        for (int i = 0; i < num; i++) {
            sum += (p1[i] * p2[i]);
        }
    } else {
        float sum = 0.f;
        // NEON ACC
        float32x4_t a_float_x4, b_float_x4;
        float32x4_t sum_float_x4;
        sum_float_x4 = vdupq_n_f32(0);
        for (int i = 0; i < num; i += 4) {
            a_float_x4 = vld1q_f32(p1 + i);
            b_float_x4 = vld1q_f32(p2 + i);
            sum_float_x4 = vaddq_f32(sum_float_x4, vmulq_f32(a_float_x4, b_float_x4));
        }
        sum += vgetq_lane_f32(sum_float_x4, 0);
        sum += vgetq_lane_f32(sum_float_x4, 1);
        sum += vgetq_lane_f32(sum_float_x4, 2);
        sum += vgetq_lane_f32(sum_float_x4, 3);
    }
}

void testVecMulAdd(float *p1, float *p2, float *p3, int num, int function_type) {
    if (function_type == 0) {
        for (int i = 0; i < num; i++)
            p3[i] += (p1[i] * p2[i]);
    } else {
        float32x4_t a_float_x4, b_float_x4, c_float_x4;
        for (int i = 0; i < num; i += 4) {
            a_float_x4 = vld1q_f32(p1 + i);
            b_float_x4 = vld1q_f32(p2 + i);
            c_float_x4 = vld1q_f32(p3 + i);
            c_float_x4 = vaddq_f32(c_float_x4, vmulq_f32(a_float_x4, b_float_x4));
            vst1q_f32(p3 + i, c_float_x4);
        }
    }
}

void testVecAdd(float *p1, float *p2, float *p3, int num, int function_type) {
    if (function_type == 0) {
        for (int i = 0; i < num; i++) {
            p3[i] = p1[i] + p2[i];
        }
    } else {
        float32x4_t a_float_x4, b_float_x4, c_float_x4;
        for (int i = 0; i < num; i += 4) {
            a_float_x4 = vld1q_f32(p1 + i);
            b_float_x4 = vld1q_f32(p2 + i);
            c_float_x4 = vaddq_f32(a_float_x4, b_float_x4);
            vst1q_f32(p3 + i, c_float_x4);
        }
    }
}


double testComponents_(int total_count, int test_function) {
    std::string function_name[3] = {"dotProduct", "vecMulAdd", "vecAdd"};
    void (*testFunction)(float *p1, float *p2, float *p3, int num, int function_type);
    if (test_function == 0) {
        testFunction = testDotProduct;
    } else if (test_function == 1) {
        testFunction = testVecMulAdd;
    } else {
        testFunction = testVecAdd;
    }
    printf("测试函数:   %s\n", function_name[test_function].c_str());
    printf("基准       ");
    double t1 = benchmarkComponents(total_count, 0, testFunction);
    printf("Neon加速   ");
    double t2 = benchmarkComponents(total_count, 1, testFunction);

    double efficiency = 100 * (t1 - t2) / t1;
    printf("优化效率:   %.2f%%\n\n", efficiency);
    return efficiency;
}

void testComponents(int total_count, std::ofstream& resultFile) {
    // 测试多次不同大小的图像
    printf("\n");
    printf("开始-----------------组件测试---------------开始\n");
    printf("测试数组大小: %d\n\n", TEST_ARRAY_SIZE);

    double efficiency[3] = {0};
    for (int i = 0; i < 3; i++) {
        efficiency[i] = testComponents_(total_count*1000, i);
    }

    printf("结束-----------------组件测试---------------结束\n");
    printf("\n");

    std::string function_name[3] = {"dotProduct", "vecMulAdd", "vecAdd"};
    if (resultFile.is_open()) {
        resultFile << "Components:" << std::endl;
        for (int i = 0; i < 3; i++) {
            resultFile << "  " << function_name[i] << ": " << efficiency[i] << std::endl;
        }
    }
}