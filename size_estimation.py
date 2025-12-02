import sys
import cv2
import numpy as np
from abc import ABC, abstractmethod


def list_cameras(max_index=10):
    """Find available cameras."""
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            cams.append(i)
            cap.release()
    return cams


def open_source(source=None, max_index=10):
    """Open camera or video file.
    
    If source is None, list available webcams and ask user to pick one.
    If source is an int or string, try to open it directly.
    """
    # Interactive camera selection
    if source is None:
        cams = list_cameras(max_index=max_index)
        if not cams:
            print("No webcams found. Provide a video file with --source /path/to/video.mp4")
            sys.exit(1)
        print("Available cameras:")
        for c in cams:
            print(f"  [{c}]")
        
        while True:
            try:
                idx = int(input("Select camera index: ").strip())
                if idx not in cams:
                    print("Invalid index. Pick one from the list above.")
                    continue
                cap = cv2.VideoCapture(idx)
                if cap is None or not cap.isOpened():
                    print("Failed to open camera. Try another index.")
                    continue
                return cap
            except ValueError:
                print("Please enter a valid integer index.")
    
    # Direct open (video file or specific camera index)
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print(f"Failed to open source: {source}")
        sys.exit(1)
    return cap


class CalibrationStrategy(ABC):
    """Abstract base class for different calibration methods"""
    
    @abstractmethod
    def get_pixels_per_metric(self, image):
        """Returns pixels per metric (e.g., pixels per cm)"""
        pass
    
    @abstractmethod
    def get_reference_points(self, image):
        """Returns reference points for visualization"""
        pass


class ManualReferenceCalibration(CalibrationStrategy):
    """Manual point selection for a reference object"""
    
    def __init__(self, known_width_cm):
        self.known_width_cm = known_width_cm
        self.points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
    def get_pixels_per_metric(self, image):
        """User clicks two points on reference object"""
        self.points = []
        clone = image.copy()
        
        cv2.namedWindow("Select Reference")
        cv2.setMouseCallback("Select Reference", self.mouse_callback)
        
        print(f"Click two points on the reference object ({self.known_width_cm} cm wide)")
        print("Press 'q' to cancel.")
        
        while len(self.points) < 2:
            display = clone.copy()
            for pt in self.points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.imshow("Select Reference", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow("Select Reference")
        
        if len(self.points) == 2:
            p1 = np.array(self.points[0], dtype=float)
            p2 = np.array(self.points[1], dtype=float)
            pixel_width = np.linalg.norm(p1 - p2)
            return pixel_width / self.known_width_cm
        else:
            print("Calibration cancelled or not enough points selected.")
        return None
    
    def get_reference_points(self, image):
        return self.points


class AprilTagCalibration(CalibrationStrategy):
    """AprilTag-based calibration (placeholder for future)"""
    
    def __init__(self, tag_size_cm, tag_family='tag36h11'):
        self.tag_size_cm = tag_size_cm
        self.tag_family = tag_family
        self.detector = None  # Will initialize when AprilTag library available
        
    def get_pixels_per_metric(self, image):
        """Detect AprilTag and compute pixels per metric"""
        raise NotImplementedError("AprilTag support coming soon!")
    
    def get_reference_points(self, image):
        return []


class ObjectMeasurement:
    """Main measurement class that uses any calibration strategy"""
    
    def __init__(self, calibration_strategy):
        self.calibration = calibration_strategy
        self.pixels_per_metric = None
        
    def calibrate(self, image):
        """Run calibration on image"""
        self.pixels_per_metric = self.calibration.get_pixels_per_metric(image)
        if self.pixels_per_metric:
            print(f"Calibrated: {self.pixels_per_metric:.2f} pixels per cm")
        else:
            print("Calibration failed.")
        return self.pixels_per_metric is not None
    
def measure_contour(self, contour):
    """Measure dimensions of a contour in real-world units"""
    if self.pixels_per_metric is None:
        raise ValueError("Must calibrate before measuring")
    
    # Get bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Changed from int0 - this is the proper numpy method
    
    # Calculate dimensions
    width_px = rect[1][0]
    height_px = rect[1][1]
    
    width_cm = width_px / self.pixels_per_metric
    height_cm = height_px / self.pixels_per_metric
    
    return {
        'width_cm': width_cm,
        'height_cm': height_cm,
        'box': box,
        'center': rect[0]
    }

def main():
    # Use your camera selection code
    cap = open_source(None)
    
    # Start with manual calibration
    calibration = ManualReferenceCalibration(known_width_cm=8.56)  # credit card width
    measurer = ObjectMeasurement(calibration)
    
    # Calibrate with first frame
    print("Capturing calibration frame...")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return
    
    if measurer.calibrate(frame):
        print("Calibration successful! Starting measurement stream...")
    else:
        print("Calibration failed")
        return
    
    # Trackbars for tuning detection
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Canny Low', 'Controls', 50, 255, lambda x: None)
    cv2.createTrackbar('Canny High', 'Controls', 100, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 500, 5000, lambda x: None)
    cv2.createTrackbar('Show Debug', 'Controls', 0, 1, lambda x: None)
    
    # Continuous video loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Get trackbar values
        canny_low = cv2.getTrackbarPos('Canny Low', 'Controls')
        canny_high = cv2.getTrackbarPos('Canny High', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        show_debug = cv2.getTrackbarPos('Show Debug', 'Controls')
        
        # Object detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug info
        detected_objects = 0
        
        # Draw reference points if available
        ref_points = calibration.get_reference_points(frame)
        for pt in ref_points:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            try:
                measurements = measurer.measure_contour(contour)
                detected_objects += 1
                
                # Draw bounding box
                cv2.drawContours(frame, [measurements['box']], 0, (0, 255, 0), 2)
                
                # Add text with measurements
                text = f"{measurements['width_cm']:.1f} x {measurements['height_cm']:.1f} cm"
                area_text = f"Area: {area:.0f}px"
                text_pos = (int(measurements['center'][0] - 40), int(measurements['center'][1] - 10))
                area_pos = (int(measurements['center'][0] - 40), int(measurements['center'][1] + 10))
                
                cv2.putText(frame, text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, area_text, area_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
            except Exception as e:
                print(f"Measurement error: {e}")
                continue
        
        # Display info
        info_text = f"Objects: {detected_objects} | Contours: {len(contours)} | PPM: {measurer.pixels_per_metric:.2f}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'r' to recalibrate", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Measurements', frame)
        
        # Show debug view
        if show_debug:
            cv2.imshow('Edges', edges)
        else:
            cv2.destroyWindow('Edges')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Recalibrate
            print("Recalibrating...")
            if measurer.calibrate(frame):
                print("Recalibration successful!")
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()