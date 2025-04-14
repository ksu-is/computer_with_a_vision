#importing necessary libraries
import cv2
import numpy as np

# Load the image
image_path = '/mnt/data/T-T-Parking-3-1.jpg'
img = cv2.imread(image_path)

# Resize if needed for display or processing consistency
img = cv2.resize(img, (1280, 720))



