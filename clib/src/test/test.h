//
// Created by liang on 2024/7/13.
//

#pragma once

#include <opencv2/opencv.hpp>

void testFullPipeLine(cv::Mat image, int total_count);

void testDetectFunction(unsigned char *rgb_image_data, int width, int height, int step, int total_count);