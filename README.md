# Sign Language Recognition

This project is a computer vision application designed to recognize and classify hand gestures using a webcam and a pre-trained model. It utilizes OpenCV for image processing, CVZone and MediaPipe for hand detection and gesture classification, and includes a data collection script to gather images for model training.

## Features

- **Real-Time Gesture Recognition**: Identifies hand gestures from webcam feed and displays the recognized gesture with a bounding box and label.
- **Data Collection**: Collects images of hand gestures for training purposes, helping to build gesture recognition models.

## Getting Started

### Prerequisites

- Python 3.8 or above
- Python packages: `opencv-python`, `numpy`,`cvzone`, `mediapipe` and `tensorflow` 

### Installation

1. **Clone the Repository**:

```bash
  git clone https://github.com/Reenook/Sign-Language-Recognition.git
  cd Sign-Language-Recognition
```

2. **Install Dependencies**:

Install the required packages:
 ```bash
 pip install opencv-python numpy cvzone mediapipe tensorflow
 ```

### Usage
**Hand Gesture Recognition**

To run the gesture recognition script:

1. Make sure your webcam is connected.

2. Make a folder within the project called Model where your model and labels will be stored

3. Specify path to your model and labels (if a custom model is used)

4. Adjust array with the correct labels at the correct indexes  (in case of a custom model) by opening `labels.txt` file 

5. Execute the script:
    
    ```bash
   python Main.py
    ```

7. The script will open a window showing the webcam feed with detected gestures and labels. Press `q` to exit.

### Data Collection

To collect images for training:

1. Create a directory within your python project which will hold all of your images or data and then make subfolders within that directory labeled by the sign whos image you want to capture. 

2. Specify the path to the directory and specific folder by adjusting the folder variable (i.e: Images/Peace)

3. Execute the data collection script:
   ```bash
   python Data_Collection.py
   ```

5. Press `s` to start capturing images , make sure to specify which directory those images will be saved in by changing the path in the folders variable 

## Data Collection Script Details

  - **Purpose**: Collects a dataset of hand gestures for model training.
  - **Instructions**: Perform different hand gestures, and it will save the captured images in a uniform (square) format to the specified directory.

## Model

- **Pre-Trained Model**: Model/keras_model.h5 (used for gesture classification)(trained on the images taken by the data collection script)
- **Specifications**:
    - The pre-trained model here is only trained on three signs "I Love you", "Call me" and "Peace"
- **Labels**: Model/labels.txt (contains the labels for gestures)

## Notes
- Ensure that the Model directory contains the keras_model.h5 and labels.txt files for gesture classification to work.
- Adjust the `offset` and `imgSize` variables in `Main.py` if needed for different hand sizes or webcam resolutions.
- For effective data collection, ensure consistent lighting and background conditions.
- To increase accuracy a larger dataset can be used for training 



