import cv2
import pickle

# Load and resize the image
image_path = 'T-T-Parking-3-1.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (1280, 720))
img_copy = img.copy()

# Parking spot size (adjust if needed)
w, h = 90, 160

# List to store parking positions
posList = []

# Mouse callback function
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        print(f"Added spot: ({x}, {y})")
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            if pos[0] < x < pos[0] + w and pos[1] < y < pos[1] + h:
                posList.pop(i)
                print(f"Removed spot: ({pos[0]}, {pos[1]})")
                break

# Load existing positions if they exist
try:
    with open("parking_spots.pkl", "rb") as f:
        posList = pickle.load(f)
        print("Loaded existing parking spots.")
except:
    pass

# Set up the OpenCV window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouseClick)

while True:
    img = img_copy.copy()
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+w, pos[1]+h), (255, 0, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Press 's' to save
    if key == ord('s'):
        with open("parking_spots.pkl", "wb") as f:
            pickle.dump(posList, f)
        print("✅ Parking spots saved to parking_spots.pkl")
        break

    # Press 'q' to quit without saving
    elif key == ord('q'):
        print("❌ Quit without saving.")
        break

cv2.destroyAllWindows()
