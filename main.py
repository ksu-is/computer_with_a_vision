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

# Laplacian edge detection
edges = cv2.Laplacian(blur, cv2.CV_64F)
edges = np.uint8(np.absolute(edges))

# === Analyze Each Spot ===
free_spaces = 0
print("\n=== Parking Spot Analysis ===")

for idx, (x1, y1, x2, y2) in enumerate(rectangles, 1):
    roi_edge = edges[y1:y2, x1:x2]
    roi_gray = gray[y1:y2, x1:x2]

    edge_score = cv2.countNonZero(roi_edge)
    brightness = np.mean(roi_gray)

    # === Scoring-Based Classification ===
    edge_score_weight = edge_score / 5000  # normalize
    brightness_score_weight = 1 - (brightness / 255)  # darker = higher weight

    score = edge_count / 1500 + (brightness / 255)
    if score > 1.2:
        status = "Occupied"
        color = (0, 0, 255)
    else:
        status = "Free"
        color = (0, 255, 0)
        free_spaces += 1
    # Draw and log
    print(f"{'✅' if status == 'Free' else '❌'} Spot {idx}: {status} | Edges: {edge_score} | Brightness: {brightness:.1f} | Score: {combined_score:.2f}")
    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)

# === Display Totals ===
cv2.putText(img_display, f"Free: {free_spaces}/{len(rectangles)}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

# === Show Image ===
cv2.imshow("Parking Detection (Scoring System)", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
input("Press Enter to exit...")








