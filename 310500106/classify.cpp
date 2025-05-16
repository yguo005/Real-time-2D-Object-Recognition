/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
This code implements real-time object classification using:
1. Feature vector extraction from video
2. Nearest neighbor classification using scaled Euclidean distance
3. Database matching against known objects
4. Unknown object detection using distance threshold
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "threshold_util.hpp"
#include "features.hpp"
#include "database_util.hpp"

/* Move to database_util.cpp

// Structure for database entry
struct DatabaseEntry {
    std::string label;
    std::vector<double> features;
};

// Function to load database and compute statistics
std::tuple<std::vector<DatabaseEntry>, std::vector<double>, std::vector<double>> 
loadDatabase(const std::string& filename) {
    std::vector<DatabaseEntry> database; //// 1. Create empty in-memory vector to hold database entries
    std::vector<double> means;
    std::vector<double> stddevs;
    std::ifstream file(filename); //2. Open and read the CSV file
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    //3. For each line in CSV
    // Read entries
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        // Create a temporary entry
        DatabaseEntry entry;
        
        // Parse the line into label and features
        // Read label
        std::getline(ss, entry.label, ',');
        
        // Read features
        while(std::getline(ss, value, ',')) {
            entry.features.push_back(std::stod(value));
        }
        
        // Add the entry to the in-memory database
        database.push_back(entry);
    }
    
    // Compute statistics for each feature
    if(!database.empty()) {
        int numFeatures = database[0].features.size();
        means.resize(numFeatures, 0.0);
        stddevs.resize(numFeatures, 0.0);
        
        // Calculate means
        for(int i = 0; i < numFeatures; i++) {
            double sum = 0;
            for(const auto& entry : database) {
                sum += entry.features[i];
            }
            means[i] = sum / database.size();
            
            // Calculate standard deviations
            double sumSq = 0;
            for(const auto& entry : database) {
                double diff = entry.features[i] - means[i];
                sumSq += diff * diff;
            }
            stddevs[i] = sqrt(sumSq / database.size());
        }
    }
    
    return {database, means, stddevs};
}

// Function to classify object using nearest neighbor
std::string classifyObject(const std::vector<double>& features,
                         const std::vector<DatabaseEntry>& database,
                         const std::vector<double>& stddevs,
                         double& minDist) {
    if(database.empty()) return "Unknown";
    
    minDist = std::numeric_limits<double>::max(); // Start with maximum possible distance
    std::string bestMatch = "Unknown";
    
    for(const auto& entry : database) {
        double dist = 0;
        for(size_t i = 0; i < features.size(); i++) {
            if(stddevs[i] > 0) {
                // Normalize the difference by standard deviation
                double normalizedDiff = (features[i] - entry.features[i]) / stddevs[i];
                dist += normalizedDiff * normalizedDiff;
            }
        }
        dist = sqrt(dist); //final Euclidean distance
        
        if(dist < minDist) {
            minDist = dist;
            bestMatch = entry.label;
        }
    }
    
    // If distance is too large, classify as unknown
    if(minDist > 5.0) {  // Threshold for unknown objects
        return "Unknown";
    }
    
    return bestMatch;
}
*/

int main(int argc, char *argv[]) {
    // Initialize video capture with camera 1
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(1);  
    if(!capdev->isOpened()) {
        printf("Unable to open video device 1, trying device 0...\n");
        delete capdev;
        capdev = new cv::VideoCapture(0);  // Fallback to camera 0
        if(!capdev->isOpened()) {
            printf("Unable to open any video device\n");
            return(-1);
        }
    }

    // Initialize database variables
    std::vector<DatabaseEntry> database;
    std::vector<double> means, stddevs;
    
    // Load database
    try {
        std::tie(database, means, stddevs) = loadDatabase("feature_db.csv");
        printf("Loaded database with %zu entries\n", database.size());
    } catch (const std::exception& e) {
        printf("Error loading database: %s\n", e.what());
        // Initialize with default values
        stddevs = std::vector<double>(6, 1.0);  // Default stddev of 1.0; Creates vector of size 6 (one for each feature), All stdDev initialized to 1.0: features remain unchanged, prevent divided by 0
        means = std::vector<double>(6, 0.0);    // Default mean of 0.0: no feature scaling/centering initially
    }


    // Create window
    cv::namedWindow("Classification", 1);
    
    // Parameters from regions.cpp
    const int minRegionSize = 1000;  // Minimum region size in pixels
    
    printf("\nControls:\n");
    printf("s: Save classification image\n");
    printf("q: Quit\n\n");
    
    cv::Mat frame;
    int imageCount = 0;
    
    for(;;) {
        *capdev >> frame;
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        
        // Process frame and compute features
        cv::Mat gray, binary;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        int kMeanThreshold = calculateThresholdKMeans(gray);
        cv::threshold(gray, binary, kMeanThreshold, 255, cv::THRESH_BINARY);
        
        // Clean up with morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
      
     
        // Find connected components 
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids, 8, CV_32S
        );
        
        cv::Mat output = frame.clone();
        
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

        // Only process if a region above minimum size
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
            cv::putText(output, "L2: " + resultL2 + " (" + std::to_string(int(distL2)) + ")", 
                       center + cv::Point2f(-20, -20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
                       
            cv::putText(output, "L1: " + resultL1 + " (" + std::to_string(int(distL1)) + ")", 
                       center + cv::Point2f(-20, 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
            
            // If either result is unknown, show learning prompt
            if(resultL2 == "unknown" || resultL1 == "unknown") {
                cv::putText(output, "Unknown Object - Press 'l' to learn", 
                           cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
            }
            
            cv::imshow("Video", output);
            char key = cv::waitKey(10);
            
            if(key == 'q') {
                break;
            }
            else if(key == 'l' && (resultL2 == "unknown" || resultL1 == "unknown")) {  // Learn new object
                // Get label from user
                std::string newLabel;
                printf("\nEnter label for new object: ");
                std::cin >> newLabel;
                
                // Save to database
                saveFeatureVector(features, center, newLabel);
                
                printf("Learned new object: %s\n", newLabel.c_str());
                
                // Reload database to update with new object
                std::tie(database, means, stddevs) = loadDatabase("feature_db.csv");
                printf("Database updated - now contains %zu objects\n", database.size());
            }
            if(key == 's') {
            // Save image with classification result
            std::string filename = cv::format("classification_%d.jpg", imageCount++);
            cv::imwrite(filename, output);
            printf("Saved %s\n", filename.c_str());
        }
        }
        
            
    }
    
    delete capdev;
    return(0);
}
