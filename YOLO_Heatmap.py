#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import numpy as np
import torch

# Load YOLOv5 model for human detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define a function to generate a heat map from detected humans
def generate_heatmap(image, boxes):
    # Create a black image with the same dimensions as the input image
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)

    # Add a rectangle at each box
    for box in boxes:
        x1, y1, x2, y2 = box.astype(np.int64)
        heatmap[y1:y2, x1:x2] += 1

    # Normalize the heat map
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply a color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    # Add the heat map to the input image
    result = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return result

# Open the video file
cap = cv2.VideoCapture('cctv_video.mp4')

# Loop through the frames
while True:
    # Read a frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use YOLOv5 model to detect humans in the frame
    results = model(frame, size=640)
    boxes = results.xyxy[0][:, :4].cpu().numpy()

    # Generate a heat map from the detected humans
    heat_map = generate_heatmap(frame, boxes)

    # Show the result
    cv2.imshow('Heat Map', heat_map)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




