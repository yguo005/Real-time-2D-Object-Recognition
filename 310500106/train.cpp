/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
This code implements a training system for collecting and storing object features.
It processes video input to:
1. Detect and segment objects in real-time
2. Compute feature vectors for the largest detected region
3. Enter label to save feature vector to a CSV file
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "threshold_util.hpp"
#include "features.hpp"



// Function to save feature vector to CSV
void saveFeatureVector(const std::vector<double>& features, const cv::Point2f& center, 
                      const std::string& label, const std::string& filename = "feature_db.csv") {
    bool fileExists = std::ifstream(filename).good();
    std::ofstream file(filename, std::ios::app); //append
    
    // First run: Write header if new file
    if (!fileExists) {
        file << "Label,Area,Perimeter,Circularity,AspectRatio,PercentFilled,Orientation\n";
    }
    
    file << label;
    for(const auto& feature : features) {
        file << "," << feature;
    }
    file << "\n";
    
    file.close();
}



int main(int argc, char *argv[]) {
    // Find first working camera
    printf("Checking available cameras...\n");
    int selectedCamera = -1;
    for(int i = 1; i < 10; i++) {  // start from camera 1 probably Camo
        cv::VideoCapture temp(i);
        if(temp.isOpened()) {
            printf("Camera %d is available\n", i);
            selectedCamera = i;
            printf("Selected camera %d\n", i);
            temp.release();
            break;
        }
    }

    if(selectedCamera == -1) {
        printf("No cameras available\n");
        return -1;
    }

    cv::VideoCapture *capdev = new cv::VideoCapture(selectedCamera);
    if(!capdev->isOpened()) {
        printf("Unable to open camera %d\n", selectedCamera);
        return(-1);
    }

    // Create window
    cv::namedWindow("Training Mode", 1);

    printf("\nTraining Mode Controls:\n");
    printf("1-3: Select region to save\n");
    printf("q: Quit\n\n");

    cv::Mat frame;
    
    for(;;) {
        *capdev >> frame;
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Process frame
        cv::Mat gray, binary;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        int kMeanThreshold = calculateThresholdKMeans(gray);
        cv::threshold(gray, binary, kMeanThreshold, 255, cv::THRESH_BINARY);

        // Clean up with morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

        // Connected Components Analysis
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids, 8, CV_32S
        );

        // Find top 5 largest regions
        std::vector<std::pair<int, int>> regions; // (label, area)
        for(int i = 1; i < numLabels; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area >= 1000) {
                regions.push_back({i, area});
            }
        }

        // Sort regions by area
        std::sort(regions.begin(), regions.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        // Keep only top 5
        if(regions.size() > 5) {
            regions.resize(5);
        }

        cv::Mat output = frame.clone();
        std::vector<std::vector<double>> allFeatures;
        std::vector<cv::Point2f> centers;

        // Process top 3 regions
        for(size_t i = 0; i < std::min(size_t(3), regions.size()); i++) {
            int label = regions[i].first;
            cv::Mat mask = (labels == label);
            std::vector<double> features = computeFeatureVector(mask);
            allFeatures.push_back(features);
            
            // Calculate center
            cv::Moments moments = cv::moments(mask, true);
            cv::Point2f center(moments.m10/moments.m00, moments.m01/moments.m00);
            centers.push_back(center);

            // Draw bounding box with number
            int x = stats.at<int>(label, cv::CC_STAT_LEFT);
            int y = stats.at<int>(label, cv::CC_STAT_TOP);
            int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            
            // Different colors for each region
            cv::Scalar color;
            switch(i) {
                case 0: color = cv::Scalar(0, 255, 0); break;    // Green
                case 1: color = cv::Scalar(255, 0, 0); break;    // Blue
                case 2: color = cv::Scalar(0, 0, 255); break;    // Red
                default: color = cv::Scalar(255, 255, 255);
            }
            
            cv::rectangle(output, cv::Point(x, y), cv::Point(x + width, y + height), 
                         color, 2);
            
            // Draw region number
            cv::putText(output, std::to_string(i + 1), 
                       cv::Point(x - 10, y + height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

            // Draw features for this region
            if(i == 0) {  // Show features only for the largest region
                std::vector<std::string> featureLabels = {
                    "Region " + std::to_string(i + 1) + " Features:",
                    "Area: " + std::to_string(int(features[0])),
                    "Perimeter: " + std::to_string(int(features[1])),
                    "Circularity: " + cv::format("%.3f", features[2]),
                    "Aspect Ratio: " + cv::format("%.3f", features[3]),
                    "Percent Filled: " + cv::format("%.3f", features[4]),
                    "Orientation: " + cv::format("%.3f", features[5])
                };

                int startY = 60;
                for(const auto& label : featureLabels) {
                    cv::putText(output, label, 
                               cv::Point(10, startY),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
                    startY += 25;
                }
            }
        }

        // Draw instructions
        cv::putText(output, "Press 1-3 to select region to save", 
                   cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);

        cv::imshow("Training Mode", output);

        char key = cv::waitKey(10);
        if(key == 'q') {
            break;
        } else if(key >= '1' && key <= '3') {
            int idx = key - '1';
            if(idx < regions.size()) {
                // Prompt for label
                std::string label;
                std::cout << "Enter label for region " << (idx + 1) << ": ";
                std::getline(std::cin, label);
                
                // Save feature vector
                saveFeatureVector(allFeatures[idx], centers[idx], label);
                std::cout << "Saved feature vector for '" << label << "'\n";
            }
        }
    }

    delete capdev;
    return(0);
}
