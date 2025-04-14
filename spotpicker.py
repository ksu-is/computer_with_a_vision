import cv2
import pickle

# Load image
img = cv2.imread("new_parking_lot.jpg")
if img is None:
    raise FileNotFoundError("❌ Couldn't load 'new_parking_lot.jpg'. Make sure it's in the project folder and the name is correct.")

img = cv2.resize(img, (1280, 720))
img_copy = img.copy()

# Parking spots: list of [(x1, y1, x2, y2)]
rectangles = []
drawing = False
start_point = (0, 0)

# Load saved data if it exists
try:
    with open('parking_spots.pkl', 'rb') as f:
        rectangles = pickle.load(f)
        print("Loaded saved parking spots.")
except:
    pass

# Mouse callback
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img[:] = img_copy[:]
        for rect in rectangles:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rectangles.append((start_point[0], start_point[1], end_point[0], end_point[1]))
        print(f"Added rectangle: {start_point} to {end_point}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete rectangle if clicked inside it
        for i, rect in enumerate(rectangles):
            x1, y1, x2, y2 = rect
            if x1 < x < x2 and y1 < y < y2:
                print(f"Removed rectangle: {rect}")
                rectangles.pop(i)
                break

cv2.namedWindow("Draw Parking Spots")
cv2.setMouseCallback("Draw Parking Spots", draw_rectangle)

while True:
    img[:] = img_copy[:]
    for rect in rectangles:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

    cv2.imshow("Draw Parking Spots", img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        with open("parking_spots.pkl", "wb") as f:
            pickle.dump(rectangles, f)
        print("✅ Saved rectangles to parking_spots.pkl")
        break
    elif key == ord('q'):
        print("❌ Quit without saving")
        break

cv2.destroyAllWindows()
