import cv2
import numpy as np
import pickle
import os
import sys
from datetime import datetime
from sklearn.cluster import DBSCAN

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
            print("Detecting parking spots automatically...")
            return auto_detect_parking_spots(IMAGE_PATH, spots_path)
            
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

def auto_detect_parking_spots(image_path, output_path="parking_spots.pkl"):
    """Automatically detect parking spots in an image using lines and contours"""
    print("Automatically detecting parking spots...")
    
    # Load and resize image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (1280, 720))
    img_original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blurring to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate the edges to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    min_area = 1000  # Minimum area for a parking spot
    max_area = 15000  # Maximum area for a parking spot
    potential_spots = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If it has 4 vertices, it could be a rectangle
            if len(approx) >= 4 and len(approx) <= 6:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if the shape is approximately a rectangle (aspect ratio)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5:
                    # Draw the rectangle for visualization
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add to potential spots
                    potential_spots.append((x, y, x + w, y + h))
    
    # If few spots were found, try an alternative method
    if len(potential_spots) < 5:
        print("Few spots detected with contour method, trying line detection...")
        img = img_original.copy()
        
        # Apply thresholding to create binary image
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to enhance lines
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find lines using Hough Line Transform
        lines = cv2.HoughLinesP(morph, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        
        # Create a blank image for line visualization
        line_img = np.zeros_like(img)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Dilate the lines to connect nearby lines
        line_img_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilated_lines = cv2.dilate(line_img_gray, kernel, iterations=2)
        
        # Find contours in the dilated line image
        contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and create parking spots from the contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    potential_spots.append((x, y, x + w, y + h))
    
    # If still too few spots, try a grid-based approach
    if len(potential_spots) < 8:
        print("Still few spots detected, applying a grid-based approach...")
        img = img_original.copy()
        
        # Detect parking lot area using edge density
        edges_copy = edges.copy()
        kernel = np.ones((15, 15), np.uint8)
        edge_density = cv2.dilate(edges_copy, kernel, iterations=1)
        
        # Find the regions with high edge density
        _, thresh_density = cv2.threshold(edge_density, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the parking area)
        max_area = 0
        parking_area = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                parking_area = contour
        
        if parking_area is not None:
            # Get bounding rectangle of the parking area
            x, y, w, h = cv2.boundingRect(parking_area)
            
            # Parameters for grid creation
            spot_width = w // 4  # Assume 4 spots horizontally
            spot_height = h // 5  # Assume 5 spots vertically
            
            # Create a grid of parking spots
            for row in range(5):
                for col in range(4):
                    spot_x = x + col * spot_width
                    spot_y = y + row * spot_height
                    cv2.rectangle(img, (spot_x, spot_y), 
                                 (spot_x + spot_width, spot_y + spot_height), 
                                 (0, 255, 0), 2)
                    potential_spots.append((spot_x, spot_y, 
                                          spot_x + spot_width, spot_y + spot_height))
    
    # Cluster similar rectangles and merge them
    if len(potential_spots) > 0:
        # Convert rectangles to centers for clustering
        centers = np.array([[rect[0] + (rect[2] - rect[0])//2, 
                            rect[1] + (rect[3] - rect[1])//2] 
                           for rect in potential_spots])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
        labels = clustering.labels_
        
        # Group rectangles by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(potential_spots[i])
        
        # Merge rectangles in each cluster
        merged_spots = []
        for label, rects in clusters.items():
            if len(rects) > 0:
                # Find the average dimensions
                avg_x1 = sum(rect[0] for rect in rects) // len(rects)
                avg_y1 = sum(rect[1] for rect in rects) // len(rects)
                avg_x2 = sum(rect[2] for rect in rects) // len(rects)
                avg_y2 = sum(rect[3] for rect in rects) // len(rects)
                
                merged_spots.append((avg_x1, avg_y1, avg_x2, avg_y2))
        
        # Final check: ensure we have a reasonable number of spots
        if 5 <= len(merged_spots) <= 30:
            # Save the detected spots
            with open(output_path, "wb") as f:
                pickle.dump(merged_spots, f)
            
            print(f"Detected and saved {len(merged_spots)} parking spots to {output_path}")
            
            # For visualization
            img_display = img_original.copy()
            for i, (x1, y1, x2, y2) in enumerate(merged_spots, 1):
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_display, str(i), (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Detected Parking Spots", img_display)
            cv2.waitKey(500)  # Just briefly show the spots before continuing
            cv2.destroyAllWindows()
            
            return merged_spots
        else:
            print(f"Detected {len(merged_spots)} spots, which seems unreasonable. Using grid approach instead.")
    
    # If all else fails, create a default grid of spots
    print("Using default grid of parking spots...")
    img_height, img_width = img.shape[:2]
    
    # Create a 4x5 grid of parking spots
    grid_spots = []
    cols, rows = 4, 5
    spot_width = img_width // cols
    spot_height = img_height // rows
    
    for row in range(rows):
        for col in range(cols):
            x1 = col * spot_width
            y1 = row * spot_height
            x2 = x1 + spot_width
            y2 = y1 + spot_height
            grid_spots.append((x1, y1, x2, y2))
    
    # Save the grid spots
    with open(output_path, "wb") as f:
        pickle.dump(grid_spots, f)
    
    print(f"Created a default grid with {len(grid_spots)} parking spots")
    return grid_spots

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

if __name__ == "__main__":
    print("Parking Spot Detection Program")
    print(f"Using image: {IMAGE_PATH}")
    
    # Check if the image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        sys.exit(1)
    
    # Remove existing spots file to force re-detection
    if os.path.exists(SPOTS_PATH):
        os.remove(SPOTS_PATH)
        print("Removed existing parking spots file for fresh detection")
    
    # Analyze the parking lot
    analyzer = ParkingLotAnalyzer()
    analyzer.analyze_parking_lot(IMAGE_PATH, SPOTS_PATH)