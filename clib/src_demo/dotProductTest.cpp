//
// Created by liang on 2024/7/13.
//

#include <iostream>
#include <chrono>
#include <random>
#include "arm_neon.h"


inline float dotProduct(const float *p1, const float *p2, int num) {
    float sum = 0.f;
    for (int i = 0; i < num; i++) {
        sum += (p1[i] * p2[i]);
    }
    return sum;
}

//p1 and p2 must be 512-bit aligned (16 float numbers)
inline float dotProductNeon(const float *p1, const float *p2, int num) {
    float sum = 0.f;

    float32x4_t a_float_x4, b_float_x4;
    float32x4_t sum_float_x4;

    sum_float_x4 = vdupq_n_f32(0);
    for (int i = 0; i < num; i+=4)
    {
        a_float_x4 = vld1q_f32(p1 + i);
        b_float_x4 = vld1q_f32(p2 + i);
        sum_float_x4 = vaddq_f32(sum_float_x4, vmulq_f32(a_float_x4, b_float_x4));
    }
    sum += vgetq_lane_f32(sum_float_x4, 0);
    sum += vgetq_lane_f32(sum_float_x4, 1);
    sum += vgetq_lane_f32(sum_float_x4, 2);
    sum += vgetq_lane_f32(sum_float_x4, 3);

    return sum;
}


int main() {
    // g++ -o dotProductTest dotProductTest.cpp -mfpu=neon
    //./dotProductTest
    const int numElements = 1024 * 1024; // 测试数据大小
    const int numIterations = 100; // 迭代次数

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 分配内存并对齐
    float *p1 = (float*) aligned_alloc(16, numElements * sizeof(float));
    float *p2 = (float*) aligned_alloc(16, numElements * sizeof(float));

    // 生成随机数据
    for (int i = 0; i < numElements; ++i) {
        p1[i] = dist(gen);
        p2[i] = dist(gen);
    }

    // 测试普通函数
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        volatile float result = dotProduct(p1, p2, numElements); // volatile防止编译器优化
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "普通函数耗时: " << duration << " ms" << std::endl;

    // 测试NEON优化函数
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        volatile float result = dotProductNeon(p1, p2, numElements);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "NEON优化函数耗时: " << duration << " ms" << std::endl;

    // 释放内存
    free(p1);
    free(p2);

    return 0;
}