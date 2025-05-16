/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <random>
#include "threshold_util.hpp"

// Function to generate random colors for regions
std::vector<cv::Vec3b> generateColors(int numColors) {
    std::vector<cv::Vec3b> colors(numColors);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for(int i = 0; i < numColors; i++) {
        colors[i] = cv::Vec3b(dis(gen), dis(gen), dis(gen));
    }
    colors[0] = cv::Vec3b(0, 0, 0); // Background is black
    return colors;
}

// Function to create colored region map
cv::Mat createRegionMap(const cv::Mat& labels, const std::vector<cv::Vec3b>& colors) {
    cv::Mat regionMap = cv::Mat::zeros(labels.size(), CV_8UC3);
    for(int i = 0; i < labels.rows; i++) {
        for(int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            regionMap.at<cv::Vec3b>(i, j) = colors[label];
        }
    }
    return regionMap;
}

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev = new cv::VideoCapture(0);
    if(!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // Create windows
    cv::namedWindow("Original", 1);
    cv::namedWindow("Thresholded", 1);
    cv::namedWindow("Regions", 1);

    // Parameters
    const int minRegionSize = 1000;  // Minimum region size in pixels
    const int maxRegions = 5;        // Maximum number of regions to display
    std::vector<cv::Vec3b> colors = generateColors(256);  // Random colors for regions

    printf("\nControls:\n");
    printf("s: Save images\n");
    printf("q: Quit\n\n");

    cv::Mat frame;
    for(;;) {
        *capdev >> frame;
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cv::imshow("Original", frame);

        // Convert to grayscale (required for basic thresholding operations) 
        cv::Mat gray, thresholded;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

         // Covert to threshold (convert grascale to black and white, seperate objects from background)
        int kMeanThreshold = calculateThresholdKMeans(gray);
        cv::threshold(gray, // input: grayscale image
        thresholded, // output: binary image
        kMeanThreshold, // value calculated from k-means
        255, // maximum value for pixels above threshold
        cv::THRESH_BINARY); // threshold type: binary

        // Clean up with morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, kernel);

        cv::imshow("Thresholded", thresholded);

        // Connected Components Analysis
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(
            thresholded, // input: binary (black and white) image 
            labels, // output: labeled image: Each connected component gets unique label (1, 2, 3, etc.)
            stats, // matrix:row: one row per label; column: x-coordinate of bounding box, y-coordinate, width, height, total number of pixels
            centroids, // matrix: row: one row per labe; column:  x and y coordinates of centroid
            8, // 8-way connectivity (include diagonal)
            CV_32S //output: label type 32-bit signed integer
        );

        // Filter regions by size and create region map
        std::vector<bool> validRegion(numLabels, false);
        std::vector<int> validRegionAreas;
        std::vector<cv::Point2d> validRegionCentroids;

        // Check each region
        for(int i = 1; i < numLabels; i++) {  // Skip background (label 0)
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area >= minRegionSize) {
                // Check if region touches boundary
                int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                int y = stats.at<int>(i, cv::CC_STAT_TOP);
                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                
                if(x > 0 && y > 0 && 
                   x + width < frame.cols - 1 && 
                   y + height < frame.rows - 1) {
                    validRegion[i] = true;
                    validRegionAreas.push_back(area);
                    validRegionCentroids.push_back(
                        cv::Point2d(centroids.at<double>(i,0), 
                                  centroids.at<double>(i,1))
                    );
                }
            }
        }

        // Create colored region map
        cv::Mat regionMap = createRegionMap(labels, colors);

        // Draw region information
        for(int i = 1; i < numLabels; i++) {
            if(validRegion[i]) {
                // Draw centroid
                cv::Point center(centroids.at<double>(i,0), centroids.at<double>(i,1));
                cv::circle(regionMap, center, 5, cv::Scalar(0,255,0), -1);
                
                // Draw bounding box
                cv::Rect bbox(
                    stats.at<int>(i, cv::CC_STAT_LEFT), // x coordinate
                    stats.at<int>(i, cv::CC_STAT_TOP), // y coordinate
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT)
                );
                cv::rectangle(regionMap, bbox, cv::Scalar(0,255,0), 2);
                
                // Display region size
                std::string areaText = std::to_string(stats.at<int>(i, cv::CC_STAT_AREA));
                cv::putText(regionMap, areaText, 
                           cv::Point(center.x - 20, center.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0,255,0), 1);
            }
        }

        cv::imshow("Regions", regionMap);

        // Handle keyboard input
        char key = cv::waitKey(10);
        if(key == 'q') {
            break;
        } else if(key == 's') {
            cv::imwrite("regions_original.jpg", frame);
            cv::imwrite("regions_thresholded.jpg", thresholded);
            cv::imwrite("regions.jpg", regionMap);
            printf("Images saved\n");
        }
    }

    delete capdev;
    return(0);
}
