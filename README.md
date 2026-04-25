# Gesture Recognition — Arduino Nano 33 BLE Sense Rev2 (Assignment 1)

## Student
Name: Ahmed Shoeb
Submission Date: 25 April 2026

## Hardware
- Arduino Nano 33 BLE Sense Rev2
- BMI270 IMU built-in — no external wiring needed
- Connected via USB only

## Gesture Classes
- Background (null/random movement)
- Circle (draw circle in air with hand)
- Rest (hold still)
- Shake (shake hand left-right)
- Updown (move hand up and down)

## Full Pipeline
1. Hold Arduino in hand
2. IMU reads accelerometer and gyroscope at 50Hz
3. 50 samples collected into a sliding window
4. 60 features extracted per window
5. Features normalized using StandardScaler values from training
6. TFLite model runs inference on Arduino
7. Predicted gesture printed to Serial Monitor

## Feature Extraction
Time domain per axis (6 axes x 5 features = 30):
- Mean
- Standard deviation
- RMS
- Minimum
- Maximum

Frequency domain per axis (6 axes x 5 bins = 30):
- PSD bins k=1 to 5 using full DFT

Total: 60 features per window

## Libraries Required
- Arduino_BMI270_BMM150 by Arduino
- tflite-micro-arduino-examples from TensorFlow GitHub

## How to Collect Data
1. Open arduino/data_collection/data_collection.ino
2. Change LABEL to the gesture name
3. Upload to Arduino Nano 33 BLE
4. Hold board in hand
5. Open Serial Monitor at 115200 baud
6. Perform gesture until DONE appears
7. Copy Serial Monitor output and save as gesture_name.csv

## How to Train
1. Open training/gesture_training.ipynb in Google Colab
2. Upload all CSV files from data/ folder
3. Run all cells in order
4. gesture_model.h downloads automatically
5. Copy scaler values from Cell 7 output

## How to Run Inference
1. Place gesture_model.h in arduino/inference/ folder
2. Paste scaler values into gesture_inference.ino
3. Upload to Arduino Nano 33 BLE
4. Open Serial Monitor at 115200 baud
5. Hold board in hand and perform gestures
6. Predictions appear in Serial Monitor every second

## Model Architecture
- Input: 60 features
- Dense layer: 64 neurons, ReLU activation
- Dropout: 0.3
- Dense layer: 32 neurons, ReLU activation
- Output: 5 neurons, Softmax activation
- Deployed as TFLite on Arduino Nano 33 BLE Sense Rev2

## Known Limitations
- Confidence is moderate due to limited training data (40 windows per gesture)
- Collecting more data per gesture would improve confidence significantly
- Background class prevents false positives for random movements
