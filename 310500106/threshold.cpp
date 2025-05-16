/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
This program is used to threshold the image.
It first converts the image to HSV to handle color saturation.
it darkens the highly saturated pixels.
it converts the image to grayscale.
it applies a blur to make regions more uniform.
Then it performs k-means clustering to automatically calculate the threshold.
applies morphological operations (opening, closing, gradient) to clean up the thresholded image
*/
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

// Function to perform k-means clustering for automatic thresholding
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
int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev = new cv::VideoCapture(0);
    if(!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // Create windows
    cv::namedWindow("Original", 1);
    cv::namedWindow("Preprocessed", 1);
    cv::namedWindow("Thresholded", 1);

    // Initialize parameters
    const int kernelSize = 3;  // Fixed kernel size
    int morphType = 0;  // 0=OPEN, 1=CLOSE, 2=GRADIENT

    printf("\nControls:\n");
    printf("m: Change morphological operation (and save current result)\n");
    printf("s: Save all images\n");
    printf("q: Quit\n\n");

    cv::Mat frame;
    for(;;) {
        *capdev >> frame;
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Show original
        cv::imshow("Original", frame);

        // Preprocessing steps
        cv::Mat preprocessed;
        
        // 1. Convert to HSV to handle color saturation
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsv, hsvChannels);
        
        // 2. Darken highly saturated pixels
        cv::Mat darkened = frame.clone();
        for(int i = 0; i < frame.rows; i++) {
            for(int j = 0; j < frame.cols; j++) {
                if(hsvChannels[1].at<uchar>(i,j) > 100) { // If saturation is high
                    darkened.at<cv::Vec3b>(i,j) *= 0.7;   // Darken the pixel
                }
            }
        }
        
        // 3. Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(darkened, gray, cv::COLOR_BGR2GRAY);
        
        // 4. Apply blur to make regions more uniform
        cv::GaussianBlur(gray, preprocessed, cv::Size(5,5), 0);
        
        cv::imshow("Preprocessed", preprocessed);

        // Thresholding using k-means
        cv::Mat thresholded;
        int threshold = calculateThresholdKMeans(preprocessed);
        cv::threshold(preprocessed, thresholded, threshold, 255, cv::THRESH_BINARY_INV);
        
        
        cv::imwrite("original.jpg", frame);
        cv::imwrite("preprocessed.jpg", preprocessed);
        cv::imwrite("thresholded.jpg", thresholded);
        
        // Clean up the thresholded image
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, 
            cv::Size(2*kernelSize + 1, 2*kernelSize + 1)
        );

        cv::Mat result;
        switch(morphType) {
            case 0: 
                cv::morphologyEx(thresholded, result, cv::MORPH_OPEN, kernel); 
                break;
            case 1: 
                cv::morphologyEx(thresholded, result, cv::MORPH_CLOSE, kernel); 
                break;
            case 2: 
                cv::morphologyEx(thresholded, result, cv::MORPH_GRADIENT, kernel); 
                break;
        }
        
        cv::imshow("Thresholded", result);

        // Handle keyboard input
        char key = cv::waitKey(10);
        if(key == 'q') {
            break;
        } else if(key == 's') {
            cv::imwrite("original.jpg", frame);
            cv::imwrite("preprocessed.jpg", preprocessed);
            cv::imwrite("thresholded.jpg", result);
            printf("Images saved\n");
        } else if(key == 'm') {
            // Save current morphological result
            std::string filename;
            switch(morphType) {
                case 0: filename = "morph_open.jpg"; break;
                case 1: filename = "morph_close.jpg"; break;
                case 2: filename = "morph_gradient.jpg"; break;
            }
            cv::imwrite(filename, result);
            printf("Saved %s\n", filename.c_str());

            // Cycle through morphological operations
            morphType = (morphType + 1) % 3;
            printf("Switched to morphological operation: ");
            switch(morphType) {
                case 0: printf("OPEN\n"); break;
                case 1: printf("CLOSE\n"); break;
                case 2: printf("GRADIENT\n"); break;
            }
        }
    }

    delete capdev;
    return(0);
}

