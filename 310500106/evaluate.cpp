/*
Spring CS5330
Project 3
Feb 19, 2025
Yunyu Guo
*/

/*
This file is used to evaluate the performance of the object recognition system.
It captures images from the camera and compares the true labels to the predicted labels.
It then prints the confusion matrix to the console and saves it to a file.
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "features.hpp"
#include "threshold_util.hpp"
#include "database_util.hpp"  // For loadDatabase function
#include <filesystem>
namespace fs = std::filesystem;  

// Structure for confusion matrix
struct ConfusionMatrix {
    std::map<std::string, std::map<std::string, int>> matrix;
    std::vector<std::string> labels;
    
    void addResult(const std::string& trueLabel, const std::string& predictedLabel) {
        matrix[trueLabel][predictedLabel]++;
        // Add new labels to the list if not already present
        if(std::find(labels.begin(), labels.end(), trueLabel) == labels.end()) {
            labels.push_back(trueLabel);
        }
        if(std::find(labels.begin(), labels.end(), predictedLabel) == labels.end()) {
            labels.push_back(predictedLabel);
        }
    }
    
    void printMatrix() {
        // Print header
        printf("\nConfusion Matrix:\n");
        printf("%-10s", "True\\Pred");
        for(const auto& label : labels) {
            printf("%-10s", label.c_str());
        }
        printf("\n");
        
        // Print rows
        for(const auto& trueLabel : labels) {
            printf("%-10s", trueLabel.c_str());
            for(const auto& predLabel : labels) {
                printf("%-10d", matrix[trueLabel][predLabel]);
            }
            printf("\n");
        }
    }
    
    /*
    void saveToFile(const std::string& filename) {
        std::ofstream file(filename);
        
        // Write header
        file << "True\\Pred,";
        for(const auto& label : labels) {
            file << label << ",";
        }
        file << "\n";
        
        // Write rows
        for(const auto& trueLabel : labels) {
            file << trueLabel << ",";
            for(const auto& predLabel : labels) {
                file << matrix[trueLabel][predLabel] << ",";
            }
            file << "\n";
        }
    } 
    */
};

int main(int argc, char *argv[]) {
    // 1. Load database for classification
    auto [database, means, stddevs] = loadDatabase("feature_db.csv");
    
    if(database.empty()) {
        printf("Error: Empty or missing database file\n");
        return -1;
    }
    
    // 1. Store feature vectors and labels
    std::vector<std::vector<double>> testFeatures;
    std::vector<std::string> trueLabels;
    
    //2. Define test images and true labels
    std::vector<std::string> testImages;
    
    // Process test images
    printf("Checking available cameras...\n");
    int selectedCamera = -1;
    for(int i = 1; i < 10; i++) {
        cv::VideoCapture temp(i);
        if(temp.isOpened()) {
            printf("Camera %d is available\n", i);
            selectedCamera = i;
            printf("Selected camera %d\n", i);
            temp.release();
            break;
        }
    }

    cv::VideoCapture *capdev = new cv::VideoCapture(selectedCamera);
    if(!capdev->isOpened()) {
        printf("Unable to open camera\n");
        return(-1);
    }

    // Create directory for test images
    fs::create_directory("test_images");

    cv::namedWindow("Test Capture", 1);
    
    printf("\nPress 1-2 to select and save a region's features\n");
    printf("Press 'q' to quit and see results\n");
    
    for(;;) {
        cv::Mat frame;
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
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel); // use binary image from last step
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        
        // Find connected components
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
        
        // Find largest region (excluding background)
        int maxArea = 0;
        int maxLabel = 0;
        for(int i = 1; i < numLabels; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area > maxArea) {
                maxArea = area;
                maxLabel = i;
            }
        }
        
        cv::Mat output = frame.clone();

        // Only process if found a large enough region
        if(maxArea >= 1000) {
            cv::Mat mask = (labels == maxLabel);
            std::vector<double> features = computeFeatureVector(mask);
            
            // Calculate center for drawing
            cv::Moments moments = cv::moments(mask, true);
            cv::Point2f center(moments.m10/moments.m00, moments.m01/moments.m00);
            
            // Draw bounding box and features
            drawBoundingBox(output, stats, maxLabel);
            drawFeatures(output, features, center);
            
            // Classify using both metrics
            double distL2, distL1;
            std::string resultL2 = classifyObject(features, database, stddevs, distL2, EUCLIDEAN);
            std::string resultL1 = classifyObject(features, database, stddevs, distL1, MANHATTAN);
            
            // Draw both results on the image
            cv::putText(output, "L2: " + resultL2, 
                       center + cv::Point2f(-20, -20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
               
            cv::putText(output, "L1: " + resultL1, 
                       center + cv::Point2f(-20, 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
            
            // Display instructions
            cv::putText(output, "Press 'e' to save L2 label, 'm' to save L1 label", 
                       cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
            
            cv::imshow("Test Capture", output);
            char key = cv::waitKey(10);
            
            if(key == 'q') {
                break;
            }
            else if(key == 'e') {  // Save with L2 label
                printf("\nSaving with L2 label: %s\n", resultL2.c_str());
                testFeatures.push_back(features);
                trueLabels.push_back(resultL2);
                printf("L2 prediction: %s (dist: %.2f)\n", resultL2.c_str(), distL2);
                printf("L1 prediction: %s (dist: %.2f)\n", resultL1.c_str(), distL1);
            }
            else if(key == 'm') {  // Save with L1 label
                printf("\nSaving with L1 label: %s\n", resultL1.c_str());
                testFeatures.push_back(features);
                trueLabels.push_back(resultL1);
                printf("L2 prediction: %s (dist: %.2f)\n", resultL2.c_str(), distL2);
                printf("L1 prediction: %s (dist: %.2f)\n", resultL1.c_str(), distL1);
            }
        }
    }

    delete capdev;
    
    // After the loop, evaluate both metrics separately
    if(!testFeatures.empty()) {
        // Create confusion matrices for both metrics
        ConfusionMatrix confMatrixL2, confMatrixL1;
        
        // Evaluate using both metrics
        for(size_t i = 0; i < testFeatures.size(); i++) {
            double distanceL2, distanceL1;
            std::string predictedL2 = classifyObject(testFeatures[i], database, stddevs, distanceL2, EUCLIDEAN);
            std::string predictedL1 = classifyObject(testFeatures[i], database, stddevs, distanceL1, MANHATTAN);
            
            // Add results to confusion matrices
            confMatrixL2.addResult(trueLabels[i], predictedL2);
            confMatrixL1.addResult(trueLabels[i], predictedL1);
        }
        
        printf("\nL2 (Euclidean) Confusion Matrix:\n");
        confMatrixL2.printMatrix();
        
        printf("\nL1 (Manhattan) Confusion Matrix:\n");
        confMatrixL1.printMatrix();
    }
    
    return 0;
}
