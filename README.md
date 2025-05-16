# Project 3: Real-time 2-D Object Recognition

- **Key Libraries:**
    - OpenCV Version: [e.g., OpenCV 4.x (as used in prior projects)]
   
- **Camera Setup:** webcam on a stand facing down over a white surface



## Project Overview
This project implements a real-time 2D object recognition system. The system identifies at least five chosen objects placed on a white surface, demonstrating translation, scale, and rotation invariance. The pipeline involves:
1.  Thresholding the input video/image to segment objects.
2.  Cleaning the binary image using morphological filtering.
3.  Segmenting the image into regions using connected components analysis.
4.  Computing translation, scale, and rotation invariant features (e.g., moments) for major regions.
5.  Collecting training data by associating feature vectors with object labels.
6.  Classifying new objects using a nearest-neighbor approach (or other implemented classifiers).
The system displays the recognized object's label and position on the output.

**Task(s) Implemented From Scratch:**
*(As per project requirements: 1 if working solo, 2 if working in pairs from: thresholding, morphological filtering, connected components, moment calculations)*
- [e.g., Thresholding]
- [e.g., Connected Components Analysis (if applicable)]

## Files Submitted
- `main.cpp` 
- `features.cpp`
- `features.h`
- `classifier.cpp` 
- `classifier.h` 
- `Makefile` 

- `readme.md`
- `object_database.csv` 


## Compilation and Execution

### Compilation
- If using a Makefile: `make`

- Example command for manual compilation (adjust as needed):
  `g++ -o object_recognition main.cpp features.cpp classifier.cpp \`pkg-config --cflags --libs opencv4\``

### Running the Executable
The system can be run to process live video from a webcam, a pre-recorded video file, or a directory of images.

**General Command Structure (example):**
`./object_recognition [mode] [optional: path_to_video_or_image_directory] [optional: path_to_object_db.csv]`

**Modes & Key Bindings:**


*   **Live Video Mode (default if no mode/path specified or if mode is `live`):**
    `./object_recognition`
    `./object_recognition live`
    `./object_recognition live my_object_db.csv`
    - The program will attempt to open the default webcam.

*   **Video File Mode:**
    `./object_recognition video path/to/your/video.mp4`
    `./object_recognition video path/to/your/video.mp4 my_object_db.csv`

*   **Image Directory Mode (if implemented):**
    `./object_recognition image_dir path/to/image_folder/`
    `./object_recognition image_dir path/to/image_folder/ my_object_db.csv`
    - Processes images one by one, press a key (e.g., 'n') to advance.

**Interactive Key Bindings (during runtime):**
- **'q'**: Quit the program.
- **'t'**: Toggle display of the thresholded image window.
- **'m'**: Toggle display of the morphologically cleaned image window.
- **'r'**: Toggle display of the region map window.
- **'f'**: Toggle display of feature values/bounding boxes on the main output.
- **'n'**: **Training Mode - New Object:**
    - Prompts the user to enter a label for the currently detected (largest/central) object.
    - Computes features for this object and stores them with the label in the object database file (e.g., `object_database.csv`).
- **'c'**: **Toggle Classification Mode:** Turns on/off the display of classified labels on objects.

- **[Key for Task 9 - Toggle Second Classifier]:**
    - E.g., "Press 'k' to switch to K-Nearest Neighbor classifier."
    - E.g., "Press 'd' to switch to Decision Tree classifier."
    - E.g., "Press 's' to switch to SSD on oriented bounding box pixels."


**Object Database File:**
- The system loads/saves object features and labels to `object_database.csv` (or the filename specified/hardcoded).
- Format: [e.g., `label,feature1,feature2,feature3,...`]
