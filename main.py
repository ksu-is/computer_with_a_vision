import cv2
import numpy as np

print("Loading parking lot image...")

img = cv2.imread("T-T-Parking-3-1.jpg")
if img is None:
    raise FileNotFoundError("❌ Image not found. Make sure the image is in the same folder as this script.")

img = cv2.resize(img, (1280, 720))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 16)
median = cv2.medianBlur(thresh, 5)
dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

# Define some example parking spots
parking_spots = [
    (100, 550, 90, 160),
    (210, 550, 90, 160),
    (320, 550, 90, 160),
    (430, 550, 90, 160),
    (540, 550, 90, 160),
    (650, 550, 90, 160),
    (760, 550, 90, 160),
    (870, 550, 90, 160),
]

free_spaces = 0
print("=== Parking Spot Analysis ===")

for idx, (x, y, w, h) in enumerate(parking_spots, 1):
    roi = dilate[y:y+h, x:x+w]
    count = cv2.countNonZero(roi)

    if count < 900:
        status = "Free"
        color = (0, 255, 0)
        free_spaces += 1
    else:
        status = "Occupied"
        color = (0, 0, 255)

    print(f"Spot {idx}: {status} (white pixels: {count})")
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

cv2.putText(img, f"Free: {free_spaces}/{len(parking_spots)}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

cv2.imshow("Parking Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

input("Press Enter to exit...")




