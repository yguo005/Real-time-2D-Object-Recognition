#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

// Function declarations
std::vector<double> computeFeatureVector(const cv::Mat& binary);
void drawFeatures(cv::Mat& image, const std::vector<double>& features, const cv::Point2f& center);
void displayFeatureVector(cv::Mat& image, const std::vector<double>& features);
void drawBoundingBox(cv::Mat& image, 
                    const cv::Mat& stats, 
                    int label, 
                    const cv::Scalar& color = cv::Scalar(0, 255, 0),
                    int thickness = 2);

#endif 
