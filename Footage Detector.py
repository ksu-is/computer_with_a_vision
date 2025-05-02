import cv2
import numpy as np
import pickle
import os
import sys
import argparse
from datetime import datetime

# Default paths for your project (can be overridden with command line arguments)
VIDEO_PATH = "Parking Footage.mp4"
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
    
    def load_frame(self, frame):
        """Resize a video frame"""
        return cv2.resize(frame, self.config['resize_dimensions'])
        
    def load_parking_spots(self, spots_path):
        """Load the parking spot coordinates"""
        if not os.path.exists(spots_path):
            print(f"Parking spots file not found: {spots_path}")
            print("Running spot definition tool first...")
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
    
    def analyze_video(self, video_path, spots_path, playback_speed=1.0, screenshot_interval=5):
        """Analyze parking lot from video footage
        
        Args:
            video_path: Path to the video file
            spots_path: Path to the parking spots definition file
            playback_speed: Speed factor (1.0 is normal, 0.5 is half speed, etc.)
            screenshot_interval: Interval in seconds to save screenshots
        """
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create screenshots directory if it doesn't exist
        screenshots_dir = "parking_screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
            print(f"Created screenshots directory: {screenshots_dir}")
        
        # Load parking spots
        rectangles = self.load_parking_spots(spots_path)
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video stream: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate delay between frames (in ms) based on playback speed
        frame_delay = int(1000 / (fps * playback_speed))
        
        # Calculate frames between screenshots
        frames_per_screenshot = int(fps * screenshot_interval)
        
        print(f"Video loaded: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Playback speed: {playback_speed}x (delay: {frame_delay}ms per frame)")
        print(f"Saving screenshots every {screenshot_interval} seconds")
        
        # For temporal smoothing - keep track of recent spot statuses
        spot_history = [[] for _ in range(len(rectangles))]
        history_length = 5  # Number of frames to keep in history
        
        # Process video frame by frame
        frame_count = 0
        time_of_last_screenshot = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = frame_count / fps
            
            # Process every 3rd frame for better performance while maintaining accuracy
            if frame_count % 3 != 0:
                continue
                
            # Resize frame
            frame = self.load_frame(frame)
            
            # Process frame
            processed_images = self.preprocess_image(frame)
            
            # Analyze each spot
            spots_data = []
            for i, coords in enumerate(rectangles):
                spot_data = self.analyze_spot(processed_images, coords)
                
                # Add to history for temporal smoothing
                if len(spot_history[i]) >= history_length:
                    spot_history[i].pop(0)  # Remove oldest data
                spot_history[i].append(spot_data['occupied'])
                
                # Apply temporal smoothing (majority vote)
                if len(spot_history[i]) >= 3:  # Need at least 3 frames for smoothing
                    # Count occupied vs free in history
                    occupied_count = sum(1 for x in spot_history[i] if x)
                    free_count = len(spot_history[i]) - occupied_count
                    
                    # Override current status with majority vote
                    spot_data['occupied'] = occupied_count > free_count
                
                spots_data.append(spot_data)
            
            # Visualize results
            result_frame = self.visualize_results(frame, spots_data, rectangles)
            
            # Add video time display
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            cv2.putText(result_frame, f"Video Time: {minutes:02d}:{seconds:02d}", (30, 160),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame number display
            cv2.putText(result_frame, f"Frame: {frame_count}", (30, 200),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add playback speed indicator
            cv2.putText(result_frame, f"Speed: {playback_speed}x", (30, 240),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save screenshot at regular intervals
            if current_time - time_of_last_screenshot >= screenshot_interval:
                screenshot_path = f"{screenshots_dir}/parking_{int(current_time)}s.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                print(f"Saved screenshot: {screenshot_path}")
                time_of_last_screenshot = current_time
            
            # Display result
            cv2.imshow("Parking Lot Analysis", result_frame)
            
            # Handle keyboard controls
            key = cv2.waitKey(frame_delay) & 0xFF
            
            # Controls:
            # q - quit
            # + - increase speed
            # - - decrease speed
            # s - take screenshot
            # p - pause/play
            
            if key == ord('q'):
                break
            elif key == ord('+'):
                playback_speed = min(playback_speed + 0.25, 4.0)
                frame_delay = int(1000 / (fps * playback_speed))
                print(f"Playback speed: {playback_speed}x (delay: {frame_delay}ms)")
            elif key == ord('-'):
                playback_speed = max(playback_speed - 0.25, 0.25)
                frame_delay = int(1000 / (fps * playback_speed))
                print(f"Playback speed: {playback_speed}x (delay: {frame_delay}ms)")
            elif key == ord('s'):
                screenshot_path = f"{screenshots_dir}/parking_manual_{int(current_time)}s.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                print(f"Manual screenshot saved: {screenshot_path}")
            elif key == ord('p'):
                # Pause - wait for another 'p' press
                print("Paused. Press 'p' to resume.")
                while True:
                    if cv2.waitKey(100) & 0xFF == ord('p'):
                        print("Resumed playback.")
                        break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Video analysis complete. Processed {frame_count} frames.")
        print(f"Screenshots saved to {screenshots_dir}/")
        
        return spots_data

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
    print(f"Created sample parking spots file at {SPOTS_PATH}")
    return sample_spots

def extract_first_frame(video_path, output_path="first_frame.jpg"):
    """Extract first frame from video for spot definition"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video stream: {video_path}")
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Extracted first frame to {output_path}")
    else:
        raise Exception("Failed to extract first frame from video")
    
    cap.release()
    return output_path

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parking Spot Detection from Video')
    parser.add_argument('--video', default=VIDEO_PATH, help='Path to the video file')
    parser.add_argument('--image', default=IMAGE_PATH, help='Path to reference image for defining spots')
    parser.add_argument('--spots', default=SPOTS_PATH, help='Path to parking spots definition file')
    parser.add_argument('--define-spots', action='store_true', help='Force redefining parking spots')
    parser.add_argument('--use-sample', action='store_true', help='Use sample parking spots')
    parser.add_argument('--speed', type=float, default=1.0, choices=[0.25, 0.5, 1.0, 2.0], 
                        help='Playback speed (0.25, 0.5, 1.0, or 2.0)')
    parser.add_argument('--screenshot-interval', type=int, default=5, 
                        help='Interval between automatic screenshots (seconds)')
    
    args = parser.parse_args()
    
    # Update paths from command line arguments
    VIDEO_PATH = args.video
    IMAGE_PATH = args.image
    SPOTS_PATH = args.spots
    
    print("Parking Spot Detection Program - Video Edition")
    print(f"Using video: {VIDEO_PATH}")
    
    # Check if the video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found!")
        sys.exit(1)
    
    # Extract first frame for spot definition if needed
    first_frame_path = "first_frame.jpg"
    if not os.path.exists(IMAGE_PATH):
        print(f"Reference image '{IMAGE_PATH}' not found.")
        print("Extracting first frame from video...")
        first_frame_path = extract_first_frame(VIDEO_PATH, first_frame_path)
        IMAGE_PATH = first_frame_path
    
    # Handle parking spots definition
    if args.define_spots or not os.path.exists(SPOTS_PATH):
        if args.use_sample:
            print("Using sample parking spots.")
            create_sample_parking_spots()
        else:
            print("Running spot definition tool...")
            define_parking_spots(IMAGE_PATH, SPOTS_PATH)
    
    # Show settings before starting
    print(f"\nSettings:")
    print(f"- Playback speed: {args.speed}x")
    print(f"- Screenshot interval: {args.screenshot_interval} seconds")
    print("\nControls:")
    print("  q - Quit")
    print("  + - Increase speed")
    print("  - - Decrease speed")
    print("  s - Take manual screenshot")
    print("  p - Pause/Resume")
    print("\nStarting analysis...")
    
    # Analyze the parking lot from video
    analyzer = ParkingLotAnalyzer()
    analyzer.analyze_video(VIDEO_PATH, SPOTS_PATH, args.speed, args.screenshot_interval)