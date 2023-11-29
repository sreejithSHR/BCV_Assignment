# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:00:26 2023

@author: Mithun
"""
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import Counter

app = Flask(__name__)

# Load the saved model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define an endpoint for the pose classification API
@app.route('/classify-pose', methods=['POST'])
def classify_pose():
    # Get the uploaded video file
    file = request.files['file']
    
    # Load the video file using OpenCV and extract the number of frames
    cap = cv2.VideoCapture(file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define an empty DataFrame to store the x, y, z, and visibility data
    columns = ['Frame', 'Joint', 'X', 'Y', 'Z', 'Visibility']
    df = pd.DataFrame(columns=columns)

    # Iterate through each frame of the video and use MediaPipe to detect the pose landmarks
    for frame in range(num_frames):
        ret, image = cap.read()
        if not ret:
            break

        # Convert the image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Extract the landmark data and store it in the DataFrame
        if results.pose_landmarks:
            landmarks = [[frame, i, lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(results.pose_landmarks.landmark)]
            df = df.append(pd.DataFrame(landmarks, columns=columns), ignore_index=True)

    # Make predictions using the KNN model
    test_set = df.iloc[:, 2:].values
    ypred = model.predict(test_set)

    # Get the counts of each predicted label
    counts = Counter(ypred)
    
    # Return the counts as a JSON response
    return jsonify(counts)

if __name__ == '__main__':
    app.run(debug=True)

