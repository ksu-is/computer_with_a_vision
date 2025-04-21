import cv2
import numpy as np
import pickle
import os
import argparse
from datetime import datetime
import time

class ParkingDetector:
    def __init__(self, config=None):
        self.config = config or {
            'edge_weight': 0.6,
            'brightness_weight': 0.2,
            'saturation_weight': 0.2,
            'occupation_threshold': 0.55,
            'resize_dimensions': (1280, 720),
            'show_details': True,
            'detection_interval': 15,  # Process every N frames
            'spot_detection_sensitivity': 150,  # For auto spot detection
            'min_spot_area': 1000,  # Minimum area of a parking spot
            'max_spot_area': 15000,  # Maximum area of a parking spot
        }
        self.spots = None
        self.frame_count = 0
        
    def load_spots(self, spots_path):
        """Load parking spot coordinates from file"""
        if os.path.exists(spots_path):
            with open(spots_path, "rb") as f:
                self.spots = pickle.load(f)
                print(f"Loaded {len(self.spots)} parking spots")
                return True
        return False
    
    def save_spots(self, spots_path):
        """Save parking spot coordinates to file"""
        with open(spots_path, "wb") as f:
            pickle.dump(self.spots, f)
            print(f"Saved {len(self.spots)} parking spots to {spots_path}")
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for analysis"""
        # Resize
        frame = cv2.resize(frame, self.config['resize_dimensions'])
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply blur
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # Edge detection (combining Canny and Laplacian)
        edges_lap = cv2.Laplacian(blur, cv2.CV_64F)
        edges_lap = np.uint8(np.absolute(edges_lap))
        
        edges_canny = cv2.Canny(blur, 50, 150)
        edges = cv2.addWeighted(edges_lap, 0.5, edges_canny, 0.5, 0)
        
        return {
            'original': frame,
            'gray': gray,
            'hsv': hsv,
            'blur': blur,
            'edges': edges
        }
    
    def auto_detect_spots(self, frame):
        """Automatically detect parking spots using image processing"""
        processed = self.preprocess_frame(frame)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            processed['blur'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 19, 2
        )
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential parking spots
        potential_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.config['min_spot_area'] < area < self.config['max_spot_area']:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (parking spots are usually rectangular)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.5:  # Reasonable aspect ratio for parking spots
                    potential_spots.append((x, y, x + w, y + h))
        
        # Further filtering to remove overlapping spots
        filtered_spots = []
        for spot in potential_spots:
            # Check if this spot overlaps significantly with any already selected spot
            overlapping = False
            for selected_spot in filtered_spots:
                overlap = self.calculate_overlap(spot, selected_spot)
                if overlap > 0.5:  # If overlap is more than 50%
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_spots.append(spot)
        
        print(f"Detected {len(filtered_spots)} potential parking spots")
        
        # Sort spots based on position (left to right, top to bottom)
        filtered_spots.sort(key=lambda spot: (spot[1] // 50, spot[0]))
        
        self.spots = filtered_spots
        return filtered_spots
    
    def calculate_overlap(self, rect1, rect2):
        """Calculate overlap ratio between two rectangles"""
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return float(intersection_area) / union_area
        
    def analyze_spot(self, processed_images, spot_coords):
        """Analyze a single parking spot"""
        x1, y1, x2, y2 = spot_coords
        
        # Extract ROIs
        roi_edge = processed_images['edges'][y1:y2, x1:x2]
        roi_gray = processed_images['gray'][y1:y2, x1:x2]
        roi_hsv = processed_images['hsv'][y1:y2, x1:x2]
        
        # Extract features
        edge_density = cv2.countNonZero(roi_edge) / max(roi_edge.shape[0] * roi_edge.shape[1], 1)
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
    
    def visualize_results(self, frame, spots_data):
        """Visualize detection results on the frame"""
        img_display = frame.copy()
        
        # Count free spots
        free_spaces = sum(1 for data in spots_data if not data['occupied'])
        total_spaces = len(spots_data)
        
        # Draw rectangles and add information
        for idx, (rect, data) in enumerate(zip(self.spots, spots_data), 1):
            x1, y1, x2, y2 = rect
            
            # Color based on occupancy
            color = (0, 0, 255) if data['occupied'] else (0, 255, 0)  # Green for free, Red for occupied
            
            # Draw rectangle
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Add spot number
            cv2.putText(img_display, f"{idx}", (x1 + 5, y1 + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add score (optional for debugging)
            if self.config['show_details']:
                cv2.putText(img_display, f"{data['score']:.2f}", (x1 + 5, y2 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
    
    def manual_define_spots(self, frame):
        """Interactive tool to manually define parking spots"""
        rectangles = []
        drawing = False
        start_point = None
        img = frame.copy()
        img_copy = img.copy()
        
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
        
        # Setup window and callback
        cv2.namedWindow("Define Parking Spots")
        cv2.setMouseCallback("Define Parking Spots", mouse_callback)
        cv2.imshow("Define Parking Spots", img)
        
        print("\n=== Manual Parking Spot Definition Tool ===")
        print("Click and drag to define parking spots")
        print("Press 's' to save and exit")
        print("Press 'c' to clear all spots")
        print("Press 'r' to remove the last spot")
        print("Press 'q' to quit without saving")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                self.spots = rectangles
                print(f"Defined {len(rectangles)} parking spots")
                break
                
            elif key == ord('c'):  # Clear all
                rectangles = []
                img = frame.copy()
                img_copy = img.copy()
                cv2.imshow("Define Parking Spots", img)
                print("Cleared all parking spots")
                
            elif key == ord('r') and rectangles:  # Remove last
                rectangles.pop()
                img = frame.copy()
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
    
    def process_video(self, video_path, spots_path, output_path=None):
        """Process video for parking spot detection and analysis"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up video writer if output is requested
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, 
                self.config['resize_dimensions']
            )
        
        # Initialize spots detection
        spots_setup_done = False
        self.frame_count = 0
        last_analysis_time = time.time()
        spots_data = []
        
        print(f"\nProcessing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Resize frame to working resolution
            frame = cv2.resize(frame, self.config['resize_dimensions'])
            self.frame_count += 1
            
            # First-time setup for spot detection
            if not spots_setup_done:
                # Try to load spots first
                if not self.load_spots(spots_path):
                    # If loading fails, offer manual or auto detection
                    print("\nParking spots not found. Please choose an option:")
                    print("1. Automatically detect parking spots")
                    print("2. Manually define parking spots")
                    choice = input("Enter your choice (1-2): ")
                    
                    if choice == '1':
                        self.auto_detect_spots(frame)
                    else:
                        self.manual_define_spots(frame)
                    
                    # Save the defined spots
                    self.save_spots(spots_path)
                
                spots_setup_done = True
                
                # Initial analysis
                processed = self.preprocess_frame(frame)
                spots_data = [self.analyze_spot(processed, spot) for spot in self.spots]
            
            # Process every N frames to save computing power
            current_time = time.time()
            time_elapsed = current_time - last_analysis_time
            
            if time_elapsed >= 1.0 / self.config['detection_interval']:
                # Process frame
                processed = self.preprocess_frame(frame)
                spots_data = [self.analyze_spot(processed, spot) for spot in self.spots]
                last_analysis_time = current_time
                
                # Print analysis results occasionally
                if self.frame_count % (fps * 5) == 0 and self.config['show_details']:  # Every 5 seconds
                    self.print_analysis(spots_data)
            
            # Visualize results
            result_frame = self.visualize_results(frame, spots_data)
            
            # Display the frame
            cv2.imshow("Parking Spot Detection", result_frame)
            
            # Write frame to output video if requested
            if video_writer:
                video_writer.write(result_frame)
            
            # Check for user exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted processing")
                break
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {self.frame_count} frames")
        if output_path:
            print(f"Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Parking Spot Detection from Video")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--spots", "-s", default="parking_spots.pkl", help="Path to save/load parking spots")
    parser.add_argument("--output", "-o", help="Path to save output video (optional)")
    parser.add_argument("--interval", "-i", type=int, default=15, help="Frame processing interval")
    parser.add_argument("--details", "-d", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Configure and run detector
    config = {
        'edge_weight': 0.6,
        'brightness_weight': 0.2,
        'saturation_weight': 0.2,
        'occupation_threshold': 0.55,
        'resize_dimensions': (1280, 720),
        'show_details': args.details,
        'detection_interval': args.interval
    }
    
    detector = ParkingDetector(config)
    detector.process_video(args.video, args.spots, args.output)

if __name__ == "__main__":
    main()