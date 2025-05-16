/*
CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025
*/
/*
Implements real-time region feature detection from video input.
It processes each frame to:
1. Convert to grayscale and apply thresholding to separate objects from background
2. Use connected components analysis to identify distinct regions
3. Compute a set of features for the largest region:
   - Area 
   - Perimeter 
   - Circularity 
   - Aspect Ratio 
   - Percent Filled 
   - Orientation 
   - Center of Mass 
*/

#include "features.hpp"
#include "threshold_util.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <ctime>
#include <string>

/* Move RegioinFeatures struct to features.hpp
// Structure to hold feature vector
struct RegionFeatures {
    double area;          // 1. Total number of pixels (m00)
    double perimeter;     // 2. Length of region boundary
    double circularity;   // 3. roundness measure
    double aspectRatio;   // 4. min(width,height)/max(width,height) of bounding box
    double percentFilled; // 5. area/bounding_box_area (density measure)
    double orientation;   // 6. Angle of least central moment
    cv::Point2f center;   // 7,8. (x,y) center of mass from m10/m00, m01/m00
    cv::RotatedRect orientedBoundingBox;  // Minimum area rectangle
};
*/


/* Move the function implementation to features_util.cpp

// Function to compute region features
RegionFeatures computeFeatures(const cv::Mat& binary, int regionId) {
    RegionFeatures features;
    
    // Find connected components: 1. identyfy distinct regions 2. label each pixel with its region ID 3. get statistics for each region
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
    
    // 1. Center of mass (translation invariant)
    features.center = cv::Point2f(moments.m10/moments.m00, moments.m01/moments.m00);
    
    // 2. Orientation (translation, rotation invariant)
    double mu11 = moments.mu11;
    double mu20 = moments.mu20;
    double mu02 = moments.mu02;
    features.orientation = 0.5 * atan2(2*mu11, mu20 - mu02);
    
    // 3. Area (translation, scale, rotation invariant)
    features.area = moments.m00;
    
    // Get bounding box information from stats
    int width = stats.at<int>(maxLabel, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(maxLabel, cv::CC_STAT_HEIGHT);
    
    // 4. Calculate perimeter  (scale dependent)
    features.perimeter = 2 * (width + height);
    
    // 5. Circularity (translation, scale, rotation invariant)
    features.circularity = 4 * M_PI * features.area / (features.perimeter * features.perimeter);
    
    // Find oriented bounding box using moments
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(mask, nonZeroPoints);
    if (!nonZeroPoints.empty()) {
        features.orientedBoundingBox = cv::minAreaRect(nonZeroPoints);
        
        // 6. Aspect ratio (translation, rotation invariant): aspectRatio = std::min(width, height) / std::max(width, height)
        features.aspectRatio = std::min(features.orientedBoundingBox.size.width, 
                                      features.orientedBoundingBox.size.height) /
                              std::max(features.orientedBoundingBox.size.width, 
                                     features.orientedBoundingBox.size.height);
        
        // 7. Percent Filled (Translation, rotation invariant)
        double boxArea = features.orientedBoundingBox.size.width * 
                        features.orientedBoundingBox.size.height;
        features.percentFilled = (boxArea > 0) ? (features.area / boxArea) : 0;
    } else {
        features.aspectRatio = 1.0; // Default if no points found
    }
    
    return features;
}

// Function to display complete feature vector
void displayFeatureVector(cv::Mat& image, const RegionFeatures& features) {
    std::vector<std::string> featureDescriptions = {
        cv::format("Feature Vector for Region:"),
        cv::format("1. Area: %.0f px", features.area),
        cv::format("2. Perimeter: %.0f px", features.perimeter),
        cv::format("3. Circularity: %.3f", features.circularity),
        cv::format("4. Aspect Ratio: %.3f", features.aspectRatio),
        cv::format("5. Percent Filled: %.3f", features.percentFilled),
        cv::format("6. Orientation: %.3f rad", features.orientation),
        cv::format("7,8. Center: (%.1f, %.1f)", features.center.x, features.center.y)
    };

    // Display feature vector on image
    int y = 30;
    for(const auto& desc : featureDescriptions) {
        cv::putText(image, desc, cv::Point(10, y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        y += 20;
    }
}

// Update drawFeatures to include complete vector display
void drawFeatures(cv::Mat& image, const RegionFeatures& features) {
    // Draw visual features
    // Center point
    cv::circle(image, features.center, 5, cv::Scalar(0,255,0), -1);
    
    // Orientation line
    double lineLength = 50.0;
    cv::Point2f endPoint(features.center.x + lineLength * cos(features.orientation),
                        features.center.y + lineLength * sin(features.orientation));
    cv::line(image, features.center, endPoint, cv::Scalar(0,255,0), 2);
    
    // Oriented bounding box
    cv::Point2f vertices[4];
    features.orientedBoundingBox.points(vertices);
    for(int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,0,255), 2);
    }
    
    // Display complete feature vector
    displayFeatureVector(image, features);
}
*/


int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev = new cv::VideoCapture(0);
    if(!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // Get video properties
    int frame_width = capdev->get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capdev->get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capdev->get(cv::CAP_PROP_FPS);

    // Create video writer
    cv::VideoWriter features_writer;
    bool is_recording = false;

    // Create window
    cv::namedWindow("Features", 1);

    printf("\nControls:\n");
    printf("r: Start/Stop recording\n");
    printf("q: Quit\n\n");

    for(;;) {
        cv::Mat frame;
        *capdev >> frame;
        
        if(frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Convert to grayscale and threshold
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
        int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
        
        cv::Mat output = frame.clone();
        
        // Process each region
        for(int i = 1; i < numLabels; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area >= 1000) {  // Minimum size threshold
                cv::Mat mask = (labels == i);
                
                // Replace RegionFeatures with direct feature vector computation
                std::vector<double> features = computeFeatureVector(mask);
                
                // Get center point for drawing
                cv::Moments moments = cv::moments(mask, true);
                cv::Point2f center(moments.m10/moments.m00, moments.m01/moments.m00);
                
                // Draw features
                drawFeatures(output, features, center);
                
                // Display feature values
                displayFeatureVector(output, features);
            }
        }
        
        cv::imshow("Video", frame);
        cv::imshow("Processed", output);
        
        char key = cv::waitKey(10);
        if(key == 'q') {
            break;
        } else if(key == 'r') {
            if(!is_recording) {
                // Start recording
                std::string timestamp = std::to_string(time(nullptr));
                features_writer.open("features_" + timestamp + ".avi",
                                   cv::VideoWriter::fourcc('M','J','P','G'),
                                   fps, cv::Size(frame_width, frame_height));
                if(!features_writer.isOpened()) {
                    printf("Error opening video file for writing\n");
                } else {
                    is_recording = true;
                    printf("Started recording\n");
                }
            } else {
                // Stop recording
                features_writer.release();
                is_recording = false;
                printf("Stopped recording\n");
            }
        }
    }

    // Clean up
    if(is_recording) {
        features_writer.release();
    }
    delete capdev;
    return(0);
}
