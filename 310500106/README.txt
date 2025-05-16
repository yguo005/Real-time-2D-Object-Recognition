CS5330 Spring 2025
Project 3
Yunyu Guo
Feb 16, 2025

How to run:
example:
cd to build folder run: ./threshold
./rgbdThreshold

threshold.cpp
press "s" to save original, preprocessed, thresholded images
press "m" to change morphological operation: close, open, gradient and save current results

rgbdThreshold.cpp trackbar:
drag the trackbar to change the threshold values
- RGB Threshold: Adjust color-based threshold value (0-255)
- Depth Threshold %: Adjust depth cutoff point (0-100%)
- RGB Weight %: Adjust relative importance of RGB vs depth (0-100%)

features.cpp
press "r" to start/stop recording
press "q" to quit

train.cpp
press "1-5" to select a region to enter label
press "n" to save the feature vector of the selected region
press "s" to save the training image
press "q" to quit

evaluate.cpp
press "e" to save L2 label: Euclidean
press "m" to save L1 label: Manhattan

