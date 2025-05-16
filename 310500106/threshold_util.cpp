#include "threshold_util.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

int calculateThresholdKMeans(const cv::Mat& grayImage) {
    // Sample 1/16 of the pixels
    std::vector<float> samples;
    for(int i = 0; i < grayImage.rows; i += 4) {
        for(int j = 0; j < grayImage.cols; j += 4) {
            samples.push_back(grayImage.at<uchar>(i,j));
        }
    }
    
    cv::Mat samplesMat(samples);
    samplesMat.convertTo(samplesMat, CV_32F);
    
    // Perform k-means with k=2
    cv::Mat labels, centers;
    cv::kmeans(samplesMat, 2, labels, 
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    
    // Calculate threshold as midpoint between the two cluster centers
    return (centers.at<float>(0) + centers.at<float>(1)) / 2;
}
