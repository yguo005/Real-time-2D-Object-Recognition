/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 19, 2025
*/

/*
This file implements the core feature computation and visualization for object recognition:

1. computeFeatureVector: Computes a 6-dimensional feature vector for object recognition:
   - Area (translation, scale, rotation invariant)
   - Perimeter (scale dependent)
   - Circularity (translation, scale, rotation invariant)
   - Aspect Ratio (translation, rotation invariant)
   - Percent Filled (translation, rotation invariant)
   - Orientation (translation, rotation invariant)

2. drawFeatures: Visualizes object features by drawing:
   - Center point
   - Orientation line

3. displayFeatureVector: Shows numerical feature values on the image
*/

#include "features.hpp"
#include "threshold_util.hpp"
#include <opencv2/opencv.hpp>
#include "database_util.hpp"

std::vector<double> computeFeatureVector(const cv::Mat& binary) {
    std::vector<double> features(6); // 6 features
    
    // Find connected components
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(
        binary,
        labels,
        stats,
        centroids,
        8,
        CV_32S
    );
    
    // Get the largest region (excluding background)
    int maxArea = 0;
    int maxLabel = 0;
    for(int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if(area > maxArea) {
            maxArea = area;
            maxLabel = i;
        }
    }
    
    // Create mask for largest region
    cv::Mat mask = (labels == maxLabel);
    
    // Calculate moments
    cv::Moments moments = cv::moments(mask, true);
    
    // 1. Area (translation, scale, rotation invariant)
    features[0] = moments.m00;
    
    // Get bounding box information from stats
    int width = stats.at<int>(maxLabel, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(maxLabel, cv::CC_STAT_HEIGHT);
    
    // 2. Perimeter (scale dependent)
    features[1] = 2 * (width + height);
    
    // 3. Circularity (translation, scale, rotation invariant)
    features[2] = 4 * M_PI * features[0] / (features[1] * features[1]);
    
    // Find oriented bounding box using moments
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(mask, nonZeroPoints);
    if (!nonZeroPoints.empty()) {
        cv::RotatedRect box = cv::minAreaRect(nonZeroPoints);
        
        // 4. Aspect ratio (translation, rotation invariant)
        features[3] = std::min(box.size.width, box.size.height) /
                     std::max(box.size.width, box.size.height);
        
        // 5. Percent filled (translation, rotation invariant)
        double boxArea = box.size.width * box.size.height;
        features[4] = (boxArea > 0) ? (features[0] / boxArea) : 0;
    } else {
        features[3] = 1.0; // Default aspect ratio if no points found
        features[4] = 0.0; // Default percent filled
    }
    
    // 6. Orientation (translation, rotation invariant)
    double mu11 = moments.mu11;
    double mu20 = moments.mu20;
    double mu02 = moments.mu02;
    features[5] = 0.5 * atan2(2*mu11, mu20 - mu02);
    
    return features;
}

void drawFeatures(cv::Mat& image, const std::vector<double>& features, const cv::Point2f& center) {
    // Draw center point
    cv::circle(image, center, 5, cv::Scalar(0,255,0), -1);
    
    // Draw orientation line
    double orientation = features[5];
    double lineLength = 50.0;
    cv::Point2f endPoint(center.x + lineLength * cos(orientation),
                        center.y + lineLength * sin(orientation));
    cv::line(image, center, endPoint, cv::Scalar(0,255,0), 2);
}

void displayFeatureVector(cv::Mat& image, const std::vector<double>& features) {
    std::vector<std::string> featureDescriptions = {
        cv::format("Feature Vector:"),
        cv::format("1. Area: %.0f px", features[0]),
        cv::format("2. Perimeter: %.0f px", features[1]),
        cv::format("3. Circularity: %.3f", features[2]),
        cv::format("4. Aspect Ratio: %.3f", features[3]),
        cv::format("5. Percent Filled: %.3f", features[4]),
        cv::format("6. Orientation: %.3f rad", features[5])
    };

    int y = 30;
    for(const auto& desc : featureDescriptions) {
        cv::putText(image, desc, cv::Point(10, y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        y += 20;
    }
}

void drawBoundingBox(cv::Mat& image, const cv::Mat& stats, int label, 
                    const cv::Scalar& color, int thickness) {
    // Get bounding box information from stats
    int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    int y = stats.at<int>(label, cv::CC_STAT_TOP);
    int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    
    // Draw rectangle
    cv::rectangle(image, 
                 cv::Point(x, y), 
                 cv::Point(x + width, y + height), 
                 color, thickness);
}