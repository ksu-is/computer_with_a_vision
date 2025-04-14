import cv2
import numpy as np
import pickle

# Load new image
img = cv2.imread("new_parking_lot.jpg")
if img is None:
    raise FileNotFoundError("Image not found!")
img = cv2.resize(img, (1280, 720))  # Optional, adjust if needed

# Load parking rectangles
with open("parking_spots.pkl", "rb") as f:
    rectangles = pickle.load(f)

# Preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 16)
median = cv2.medianBlur(thresh, 5)
dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)





