import cv2
import numpy as np
import pickle

# === Load Image ===
img = cv2.imread("new_parking_lot.jpg")
if img is None:
    raise FileNotFoundError("Image not found.")
img = cv2.resize(img, (1280, 720))
img_display = img.copy()

# === Load Parking Spot Rectangles ===
with open("parking_spots.pkl", "rb") as f:
    rectangles = pickle.load(f)

# === Preprocess Image ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 16)
median = cv2.medianBlur(thresh, 5)
dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)
# Laplacian edge detection
edges = cv2.Laplacian(blur, cv2.CV_64F)
edges = np.uint8(np.absolute(edges))

# === Analyze Each Spot ===
free_spaces = 0
print("\n=== Parking Spot Analysis ===")

for idx, rect in enumerate(rectangles, 1):
    x1, y1, x2, y2 = rect
    roi_thresh = dilate[y1:y2, x1:x2]
    roi_orig = gray[y1:y2, x1:x2]

    edge_count = cv2.countNonZero(roi_thresh)
    brightness = np.mean(roi_orig)
    score = edge_count / 4000 + (brightness / 255)

    if score > 0.8:
        status = "Occupied"
        color = (0, 0, 255)
    else:
        status = "Free"
        color = (0, 255, 0)
        free_spaces += 1

    print(f"{'✅' if status == 'Free' else '❌'} Spot {idx}: {status} | Edges: {edge_count} | Brightness: {brightness:.1f} | Score: {score:.2f}")
    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
# === Display Totals ===
cv2.putText(img_display, f"Free: {free_spaces}/{len(rectangles)}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

# === Show Image ===
cv2.imshow("Parking Detection (Scoring System)", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
input("Press Enter to exit...")








