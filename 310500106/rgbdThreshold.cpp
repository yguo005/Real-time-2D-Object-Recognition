/* CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
  Combines RGB and Depth information for thresholding.
  It uses a weighted average of the RGB and Depth thresholds.
  It also uses morphological operations to clean up the thresholded image.
  1. RGB Thresholding:
     - Converts image to grayscale
     - Applies inverse binary threshold to detect dark objects on light background
  
  2. Depth Thresholding:
     - Uses Depth Anything (DA2) network to estimate depth
     - Thresholds based on depth values to separate foreground objects
  
  3. Combined Approach:
     - Weighted combination of RGB and depth thresholds
     - Adjustable weights to favor either RGB or depth information
     - Morphological operations (opening, closing) to clean up the result

*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "DA2Network.hpp"

class RGBDThresholder {
private:
    int rgbThreshold = 128;
    float depthThreshold = 0.5;
    float weightRGB = 0.5;    // Weight for RGB contribution
    float weightDepth = 0.5;  // Weight for depth contribution

public:
    cv::Mat combineThresholds(const cv::Mat& rgbImage, const cv::Mat& depthMap) {
        cv::Mat rgbThresholded, depthThresholded, combined;
        
        // RGB thresholding
        cv::Mat gray;
        cv::cvtColor(rgbImage, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, rgbThresholded, rgbThreshold, 255, cv::THRESH_BINARY_INV);

        // Depth thresholding
        cv::Mat depthNorm;
        cv::normalize(depthMap, depthNorm, 0, 1, cv::NORM_MINMAX);
        depthThresholded = (depthNorm < depthThreshold) * 255;

        // Combine thresholds using weights
        combined = cv::Mat::zeros(rgbImage.size(), CV_8UC1);
        for(int i = 0; i < combined.rows; i++) {
            for(int j = 0; j < combined.cols; j++) {
                float rgbVote = rgbThresholded.at<uchar>(i,j) / 255.0f;
                float depthVote = depthThresholded.at<uchar>(i,j) / 255.0f;
                float combinedVote = (rgbVote * weightRGB + depthVote * weightDepth);
                combined.at<uchar>(i,j) = (combinedVote > 0.5) ? 255 : 0;
            }
        }

        return combined;
    }

    // Setters for parameters
    void setRGBThreshold(int val) { rgbThreshold = val; }
    void setDepthThreshold(float val) { depthThreshold = val; }
    void setWeights(float rgb, float depth) {
        weightRGB = rgb;
        weightDepth = depth;
    }
};

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev = new cv::VideoCapture(0);
    if(!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // Initialize DA2Network
    DA2Network da_net("../model_fp16.onnx");

    // Get video properties
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    
    float reduction = 0.5;
    float scale_factor = 256.0 / (refS.height * reduction);

    // Create windows
    cv::namedWindow("RGB", 1);
    cv::namedWindow("Depth", 1);
    cv::namedWindow("Combined Threshold", 1);

    // Create thresholder
    RGBDThresholder thresholder;

    // Create trackbars
    int rgbThresh = 128;
    int depthThreshPercent = 50;
    int rgbWeight = 50;
    cv::createTrackbar("RGB Threshold", "Combined Threshold", &rgbThresh, 255);
    cv::createTrackbar("Depth Threshold %", "Combined Threshold", &depthThreshPercent, 100);
    cv::createTrackbar("RGB Weight %", "Combined Threshold", &rgbWeight, 100);

    cv::Mat frame;
    for(;;) {
        *capdev >> frame;
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Resize frame for speed
        cv::resize(frame, frame, cv::Size(), reduction, reduction);

        // Get depth map
        cv::Mat depthMap, depthVis;
        da_net.set_input(frame, scale_factor);
        da_net.run_network(depthMap, frame.size());
        cv::applyColorMap(depthMap, depthVis, cv::COLORMAP_INFERNO);

        // Update thresholder parameters
        thresholder.setRGBThreshold(rgbThresh);
        thresholder.setDepthThreshold(depthThreshPercent / 100.0f);
        thresholder.setWeights(rgbWeight / 100.0f, (100 - rgbWeight) / 100.0f);

        // Get combined threshold
        cv::Mat result = thresholder.combineThresholds(frame, depthMap);

        // Clean up result with morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel); // Opening removes small noise
        cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel); // Closing fills small holes

        // Show results
        cv::imshow("RGB", frame);
        cv::imshow("Depth", depthVis);
        cv::imshow("Combined Threshold", result);

        // Handle keyboard input
        char key = cv::waitKey(10);
        if(key == 'q') {
            break;
        } else if(key == 's') {
            cv::imwrite("rgb.jpg", frame);
            cv::imwrite("depth.jpg", depthVis);
            cv::imwrite("combined.jpg", result);
            printf("Images saved with weights: RGB %d%%, Depth %d%%\n", 
                   rgbWeight, (100 - rgbWeight));
        }
    }

    delete capdev;
    return(0);
}
