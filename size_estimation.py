import cv2
import numpy as np
import sys
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


def open_source(source):
    """Open camera or video file."""
    if source is None:
        cams = list_cameras()
        if not cams:
            print("No webcams found. Provide a video file with --source /path/to/video.mp4")
            sys.exit(1)
        print("Available cameras:")
        for c in cams:
            print(f"  [{c}]")
        idx = int(input("Select camera index: ").strip())
        cap = cv2.VideoCapture(idx)
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
        
        cv2.namedWindow("Select Reference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Reference", 1280, 720)
        cv2.setMouseCallback("Select Reference", self.mouse_callback)
        
        print(f"Click two points on the reference object ({self.known_width_cm} cm wide)")
        
        while len(self.points) < 2:
            display = clone.copy()
            for pt in self.points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.imshow("Select Reference", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow("Select Reference")
        
        if len(self.points) == 2:
            pixel_width = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
            return pixel_width / self.known_width_cm
        return None
    
    def get_reference_points(self, image):
        return self.points


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
        return self.pixels_per_metric is not None
    
    def measure_contour(self, contour):
        """Measure dimensions of a contour in real-world units"""
        if self.pixels_per_metric is None:
            raise ValueError("Must calibrate before measuring")
        
        # Get bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Proper numpy integer conversion
        
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
    
    # Set camera resolution (helps with window size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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
    
    # Create resizable windows
    cv2.namedWindow('Measurements', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Measurements', 1280, 720)
    cv2.namedWindow('Controls')
    
    # Trackbars for tuning detection
    cv2.createTrackbar('Canny Low', 'Controls', 50, 255, lambda x: None)
    cv2.createTrackbar('Canny High', 'Controls', 150, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 1000, 10000, lambda x: None)
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
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)
        
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
                text_pos = (int(measurements['center'][0] - 60), int(measurements['center'][1] - 15))
                area_pos = (int(measurements['center'][0] - 60), int(measurements['center'][1] + 10))
                
                # Black background for text
                cv2.rectangle(frame, 
                            (text_pos[0]-5, text_pos[1]-20), 
                            (text_pos[0]+200, area_pos[1]+5), 
                            (0,0,0), -1)
                
                cv2.putText(frame, text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, area_text, area_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
            except Exception as e:
                print(f"Measurement error: {e}")
                continue
        
        # Display info with background
        cv2.rectangle(frame, (5, 5), (500, 90), (0, 0, 0), -1)
        info_text = f"Objects: {detected_objects} | Contours: {len(contours)}"
        ppm_text = f"PPM: {measurer.pixels_per_metric:.2f}"
        help_text = "q=quit | r=recalibrate"
        
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, ppm_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, help_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Measurements', frame)
        
        # Show debug view
        if show_debug:
            cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Edges', 640, 480)
            cv2.imshow('Edges', edges)
        else:
            try:
                cv2.destroyWindow('Edges')
            except:
                pass
        
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