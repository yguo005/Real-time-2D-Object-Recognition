/*
Spring CS5330
Project 3
Feb 19, 2025
Yunyu Guo
*/

/*
Database Utility Functions:
This file implements database operations and object classification using different distance metrics:

1. loadDatabase: Loads feature vectors and labels from CSV file
   - Reads database entries (label + features)
   - Computes mean and standard deviation for feature normalization
   - Returns database entries and statistics

2. classifyObject: Performs nearest neighbor classification using two distance metrics:
   - L2 (Euclidean) distance with threshold 3.0
   - L1 (Manhattan) distance with threshold 5.0 (larger due to L1 properties)
   Both metrics use feature standardization (normalized by standard deviation)
   Returns "unknown" if distance exceeds threshold
*/


#include "database_util.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <set>

/* Move to database_util.hpp

// Structure for database entry
struct DatabaseEntry {
    std::string label;
    std::vector<double> features;
};
*/

// Function to load database and compute statistics
std::tuple<std::vector<DatabaseEntry>, std::vector<double>, std::vector<double>> 
loadDatabase(const std::string& filename) {
    std::vector<DatabaseEntry> database; // 1. Create empty in-memory vector to hold database entries
    std::vector<double> means(6, 0.0);
    std::vector<double> stddevs(6, 1.0);
    
    std::ifstream file(filename); //2. Open and read the CSV file
    if (!file.is_open()) {
        printf("Database file not found: %s\n", filename.c_str());
        return std::make_tuple(database, means, stddevs);
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    printf("Header: %s\n", line.c_str());  // Debug: print header
    
    //3. For each line in CSV
    // Read entries
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label;
        std::vector<double> features;
        
        // Get label
        std::getline(ss, label, ',');
        
        // Get features
        std::string value;
        while (std::getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }
        
        if (features.size() == 6) {  // Make sure have all features
            database.push_back({label, features});
            printf("Loaded entry: %s with %zu features\n", label.c_str(), features.size());  // Debug: print each entry
        } else {
            printf("Warning: Skipping entry with incorrect feature count: %s (%zu features)\n", 
                   label.c_str(), features.size());
        }
    }
    
    printf("\nDatabase Summary:\n");
    printf("Total entries loaded: %zu\n", database.size());
    printf("Unique labels: ");
    std::set<std::string> uniqueLabels;
    for(const auto& entry : database) {
        uniqueLabels.insert(entry.label);
    }
    for(const auto& label : uniqueLabels) {
        printf("%s, ", label.c_str());
    }
    printf("\n\n");
    
    // Compute statistics if database has entries
    if(!database.empty()) {
        // Compute means
        for(const auto& entry : database) {
            for(size_t i = 0; i < entry.features.size(); i++) {
                means[i] += entry.features[i];
            }
        }
        for(auto& mean : means) {
            mean /= database.size();
        }
        
        // Compute standard deviations
        for(const auto& entry : database) {
            for(size_t i = 0; i < entry.features.size(); i++) {
                double diff = entry.features[i] - means[i];
                stddevs[i] += diff * diff;
            }
        }
        for(auto& stddev : stddevs) {
            stddev = sqrt(stddev / (database.size() - 1));  // Use n-1 for sample stddev
        }
        
    }
    
    return std::make_tuple(database, means, stddevs);
}

// Function to classify object using nearest neighbor
std::string classifyObject(const std::vector<double>& features,
                         const std::vector<DatabaseEntry>& database,
                         const std::vector<double>& stddevs,
                         double& minDist,
                         DistanceMetric metric) {
    if(database.empty()) return "unknown";
    
    minDist = std::numeric_limits<double>::max();
    std::string bestMatch = "unknown";
    
    // Key 1: Feature weights - give more importance to reliable features
    std::vector<double> weights = {
        1.0,  // Area
        1.0,  // Perimeter
        2.0,  // Circularity (more important)
        2.0,  // Aspect Ratio (more important)
        1.5,  // Percent Filled
        0.5   // Orientation (less reliable)
    };
    
    // Key 2: Adjust thresholds for better unknown detection
    const double L2_THRESHOLD = 2.5;
    const double L1_THRESHOLD = 4.0;
    
    const double DISTANCE_THRESHOLD = (metric == EUCLIDEAN) ? L2_THRESHOLD : L1_THRESHOLD;
    
    for(const auto& entry : database) {
        double distance = 0.0;
        
        if(metric == EUCLIDEAN) {// L2 (Euclidean) distance
            for(size_t i = 0; i < features.size(); i++) {
                if(stddevs[i] > 0) {
                    double normalizedDiff = (features[i] - entry.features[i]) / stddevs[i];
                    distance += weights[i] * normalizedDiff * normalizedDiff;
                }
            }
            distance = sqrt(distance);
        } else {
            for(size_t i = 0; i < features.size(); i++) {
                if(stddevs[i] > 0) {
                    double normalizedDiff = (features[i] - entry.features[i]) / stddevs[i];
                    distance += weights[i] * std::abs(normalizedDiff);
                }
            }
        }
        
        // Key 4: Update best match only if significantly better
        if(distance < minDist) {
            minDist = distance;
            bestMatch = entry.label;
        }
    }
    
    // Key 5: Strict threshold check for unknown objects
    if(minDist > DISTANCE_THRESHOLD) {
        bestMatch = "unknown";
    }
    
    return bestMatch;
}

void saveFeatureVector(const std::vector<double>& features, 
                      const cv::Point2f& center,
                      const std::string& label, 
                      const std::string& filename) {
    bool fileExists = std::ifstream(filename).good();
    std::ofstream file(filename, std::ios::app);
    
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