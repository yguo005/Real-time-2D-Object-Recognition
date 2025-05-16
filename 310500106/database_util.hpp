#ifndef DATABASE_UTIL_H
#define DATABASE_UTIL_H

#include <vector>
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>


// Structure for database entry
struct DatabaseEntry {
    std::string label;
    std::vector<double> features;
};

enum DistanceMetric {
    EUCLIDEAN,
    MANHATTAN
};

// Function declarations
std::tuple<std::vector<DatabaseEntry>, std::vector<double>, std::vector<double>> 
loadDatabase(const std::string& filename);

std::string classifyObject(const std::vector<double>& features,
                         const std::vector<DatabaseEntry>& database,
                         const std::vector<double>& stddevs,
                         double& minDist,
                         DistanceMetric metric = EUCLIDEAN);

void saveFeatureVector(const std::vector<double>& features, 
                      const cv::Point2f& center,
                      const std::string& label, 
                      const std::string& filename = "feature_db.csv");

#endif
