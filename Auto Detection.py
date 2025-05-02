import cv2
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Fixed image path for your project
IMAGE_PATH = "new_parking_lot.jpg"
SPOTS_PATH = "parking_spots.pkl"

class ParkingLotAnalyzer:
    def __init__(self, config=None):
        self.config = config or {
            'edge_weight': 0.6,
            'brightness_weight': 0.2,
            'saturation_weight': 0.2,
            'occupation_threshold': 0.55,
            'resize_dimensions': (1280, 720),
            'show_details': True
        }
        
    def load_image(self, image_path):
        """Load and resize the input image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path)
        return cv2.resize(img, self.config['resize_dimensions'])
        
    def load_parking_spots(self, spots_path):
        """Load the parking spot coordinates"""
        if not os.path.exists(spots_path):
            print(f"Parking spots file not found: {spots_path}")
            print("Starting spot definition tool...")
            return define_parking_spots(IMAGE_PATH, spots_path)
            
        with open(spots_path, "rb") as f:
            return pickle.load(f)
    
    def preprocess_image(self, img):
        """Preprocess the image for analysis"""
        # Basic preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # Edge detection (using both Canny and Laplacian for robustness)
        edges_lap = cv2.Laplacian(blur, cv2.CV_64F)
        edges_lap = np.uint8(np.absolute(edges_lap))
        
        edges_canny = cv2.Canny(blur, 50, 150)
        edges = cv2.addWeighted(edges_lap, 0.5, edges_canny, 0.5, 0)
        
        return {
            'gray': gray,
            'hsv': hsv,
            'blur': blur,
            'edges': edges
        }
    
    def analyze_spot(self, processed_images, spot_coords):
        """Analyze a single parking spot"""
        x1, y1, x2, y2 = spot_coords
        
        # Extract ROIs
        roi_edge = processed_images['edges'][y1:y2, x1:x2]
        roi_gray = processed_images['gray'][y1:y2, x1:x2]
        roi_hsv = processed_images['hsv'][y1:y2, x1:x2]
        
        # Extract features
        edge_density = cv2.countNonZero(roi_edge) / (roi_edge.shape[0] * roi_edge.shape[1])
        brightness = np.mean(roi_gray) / 255
        saturation = np.mean(roi_hsv[:,:,1]) / 255
        
        # Calculate occupation score (weighted combination of features)
        edge_score = edge_density * self.config['edge_weight']
        brightness_score = (1 - brightness) * self.config['brightness_weight']
        saturation_score = saturation * self.config['saturation_weight']
        
        occupation_score = edge_score + brightness_score + saturation_score
        
        # Determine status
        is_occupied = occupation_score > self.config['occupation_threshold']
        
        return {
            'occupied': is_occupied,
            'edge_density': edge_density,
            'brightness': brightness,
            'saturation': saturation,
            'score': occupation_score
        }
    
    def visualize_results(self, img, spots_data, rectangles):
        """Visualize detection results on the image"""
        img_display = img.copy()
        
        # Count free spots
        free_spaces = sum(1 for data in spots_data if not data['occupied'])
        total_spaces = len(spots_data)
        
        # Draw rectangles and add information
        for idx, (rect, data) in enumerate(zip(rectangles, spots_data), 1):
            x1, y1, x2, y2 = rect
            
            # Color based on occupancy
            color = (0, 0, 255) if data['occupied'] else (0, 255, 0)  # Green for free, Red for occupied
            
            # Draw rectangle
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Add spot number
            cv2.putText(img_display, f"{idx}", (x1 + 5, y1 + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add summary information
        cv2.putText(img_display, f"Free: {free_spaces}/{total_spaces}", (30, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img_display, f"Occupancy: {(1 - free_spaces/total_spaces)*100:.1f}%", (30, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img_display, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (30, 120),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img_display
    
    def print_analysis(self, spots_data):
        """Print detailed analysis of each spot"""
        print("\n=== Parking Spot Analysis ===")
        
        for idx, data in enumerate(spots_data, 1):
            status = "Occupied" if data['occupied'] else "Free"
            status_icon = "❌" if data['occupied'] else "✅"
            
            print(f"{status_icon} Spot {idx}: {status} | " +
                  f"Edges: {data['edge_density']:.3f} | " +
                  f"Brightness: {data['brightness']:.3f} | " +
                  f"Saturation: {data['saturation']:.3f} | " +
                  f"Score: {data['score']:.3f}")
        
        # Summary
        free_spaces = sum(1 for data in spots_data if not data['occupied'])
        print(f"\nTotal spots: {len(spots_data)}")
        print(f"Free spots: {free_spaces}")
        print(f"Occupied spots: {len(spots_data) - free_spaces}")
        print(f"Occupancy rate: {((len(spots_data) - free_spaces) / len(spots_data) * 100):.1f}%")
    
    def analyze_parking_lot(self, image_path, spots_path):
        """Main function to analyze parking lot"""
        # Load data
        img = self.load_image(image_path)
        rectangles = self.load_parking_spots(spots_path)
        
        # Process image
        processed_images = self.preprocess_image(img)
        
        # Analyze each spot
        spots_data = []
        for coords in rectangles:
            spot_data = self.analyze_spot(processed_images, coords)
            spots_data.append(spot_data)
        
        # Print analysis results
        if self.config['show_details']:
            self.print_analysis(spots_data)
        
        # Visualize results
        result_image = self.visualize_results(img, spots_data, rectangles)
        
        # Display result
        cv2.imshow("Parking Spot Detection", result_image)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return spots_data, result_image

def define_parking_spots(image_path, output_path="parking_spots.pkl"):
    """Interactive tool to define parking spots on an image"""
    rectangles = []
    drawing = False
    start_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, img_copy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            img_copy = img.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_temp = img_copy.copy()
            cv2.rectangle(img_temp, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Define Parking Spots", img_temp)
            
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end_point = (x, y)
            
            # Ensure coordinates are in correct order (top-left, bottom-right)
            x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
            
            rectangles.append((x1, y1, x2, y2))
            
            # Draw the rectangle on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(len(rectangles)), (x1+5, y1+20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Define Parking Spots", img)
    
    # Load and resize image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (1280, 720))
    img_copy = img.copy()
    
    # Setup window and callback
    cv2.namedWindow("Define Parking Spots")
    cv2.setMouseCallback("Define Parking Spots", mouse_callback)
    cv2.imshow("Define Parking Spots", img)
    
    print("\n=== Parking Spot Definition Tool ===")
    print("Click and drag to define parking spots")
    print("Press 's' to save and exit")
    print("Press 'c' to clear all spots")
    print("Press 'r' to remove the last spot")
    print("Press 'q' to quit without saving")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Save
            with open(output_path, "wb") as f:
                pickle.dump(rectangles, f)
            print(f"Saved {len(rectangles)} parking spots to {output_path}")
            break
            
        elif key == ord('c'):  # Clear all
            rectangles = []
            img = cv2.imread(image_path)
            img = cv2.resize(img, (1280, 720))
            img_copy = img.copy()
            cv2.imshow("Define Parking Spots", img)
            print("Cleared all parking spots")
            
        elif key == ord('r') and rectangles:  # Remove last
            rectangles.pop()
            img = cv2.imread(image_path)
            img = cv2.resize(img, (1280, 720))
            for i, (x1, y1, x2, y2) in enumerate(rectangles, 1):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(i), (x1+5, y1+20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            img_copy = img.copy()
            cv2.imshow("Define Parking Spots", img)
            print(f"Removed last spot, {len(rectangles)} remaining")
            
        elif key == ord('q'):  # Quit without saving
            break
    
    cv2.destroyAllWindows()
    return rectangles

# Create sample parking spots if needed
def create_sample_parking_spots():
    """Create a sample parking spots file with some pre-defined spots"""
    sample_spots = [
        (100, 100, 150, 150),
        (200, 100, 250, 150),
        (300, 100, 350, 150),
        (400, 100, 450, 150),
        (100, 200, 150, 250),
        (200, 200, 250, 250),
        (300, 200, 350, 250),
        (400, 200, 450, 250)
    ]
    with open(SPOTS_PATH, "wb") as f:
        pickle.dump(sample_spots, f)
    print(f"Created sample parking spots file with 8 predefined spots")
    return sample_spots

if __name__ == "__main__":
    print("Parking Spot Detection Program")
    print(f"Using image: {IMAGE_PATH}")
    
    # Check if the image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        sys.exit(1)
    
    # Check if spots file exists, create sample if it doesn't
    if not os.path.exists(SPOTS_PATH):
        print("No parking spots defined. Creating sample parking spots...")
        create_sample_parking_spots()
    
    # Analyze the parking lot
    analyzer = ParkingLotAnalyzer()
    analyzer.analyze_parking_lot(IMAGE_PATH, SPOTS_PATH)