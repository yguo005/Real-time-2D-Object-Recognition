/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
utility function for calculating threshold using k-means clustering
*/

#ifndef THRESHOLD_UTIL_H
#define THRESHOLD_UTIL_H

#include <opencv2/opencv.hpp>

// Function declared in threshold.cpp
int calculateThresholdKMeans(const cv::Mat& grayImage);

#endif // THRESHOLD_UTIL_H
