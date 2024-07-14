//
// Created by liang on 2024/7/13.
//

#pragma once
#include <opencv2/opencv.hpp>

void testFullPipeLine(const cv::Mat &image, int total_count, std::ofstream& resultFile);

void testDetectFunction(int total_count, std::ofstream& resultFile);

void testComponents(int total_count, std::ofstream& resultFile);