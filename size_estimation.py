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
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open source: {source}")
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


class HomographyCalibration(CalibrationStrategy):
    """4-point calibration with perspective correction"""
    
    def __init__(self, ref_width_cm, ref_height_cm):
        self.ref_width_cm = ref_width_cm
        self.ref_height_cm = ref_height_cm
        self.points = []
        self.H = None       # Homography matrix: original -> top-down
        self.H_inv = None   # Inverse homography: top-down -> original
        # In warped (top-down) space we define a fixed resolution:
        self.pixels_per_cm_warped = 10  # pixels per cm in the top-down metric image
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
    def get_pixels_per_metric(self, image):
        """User clicks 4 corners of reference rectangle"""
        self.points = []
        clone = image.copy()
        
        cv2.namedWindow("Select Reference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Reference", 1280, 720)
        cv2.setMouseCallback("Select Reference", self.mouse_callback)
        
        print(f"\nClick 4 corners of reference rectangle ({self.ref_width_cm} x {self.ref_height_cm} cm)")
        print("Order: top-left, top-right, bottom-right, bottom-left")
        print("Press 'q' to cancel\n")
        
        while len(self.points) < 4:
            display = clone.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 8, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (pt[0]+15, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw lines between points
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    cv2.line(display, self.points[i], self.points[i+1], (0, 255, 0), 2)
            if len(self.points) == 4:
                cv2.line(display, self.points[3], self.points[0], (0, 255, 0), 2)
            
            # Instructions on image
            cv2.rectangle(display, (5, 5), (550, 100), (0, 0, 0), -1)
            cv2.putText(display, f"Click corner {len(self.points)+1} of 4", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Order: TL -> TR -> BR -> BL", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
            cv2.imshow("Select Reference", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow("Select Reference")
        
        if len(self.points) == 4:
            # Source points (from image)
            src_pts = np.float32(self.points)
            
            # Destination points (ideal top-down view of the card)
            width_px = self.ref_width_cm * self.pixels_per_cm_warped
            height_px = self.ref_height_cm * self.pixels_per_cm_warped
            
            dst_pts = np.float32([
                [0, 0],
                [width_px, 0],
                [width_px, height_px],
                [0, height_px]
            ])
            
            # Compute homography: original -> top-down metric space
            self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            print(f"Homography computed successfully!")
            # Our "pixels per cm" in warped metric space is fixed by design:
            return self.pixels_per_cm_warped
        
        return None
    
    def get_reference_points(self, image):
        return self.points
    
    def warp_image(self, image):
        """Warp only the card region to top-down view (mostly for debugging)."""
        if self.H is None:
            return image
        
        width_px = int(self.ref_width_cm * self.pixels_per_cm_warped)
        height_px = int(self.ref_height_cm * self.pixels_per_cm_warped)
        warped = cv2.warpPerspective(image, self.H, (width_px, height_px))
        return warped


class ObjectMeasurement:
    """Main measurement class that uses any calibration strategy"""
    
    def __init__(self, calibration_strategy):
        self.calibration = calibration_strategy
        self.pixels_per_metric = None  # in the warped metric space
        
    def calibrate(self, image):
        """Run calibration on image"""
        self.pixels_per_metric = self.calibration.get_pixels_per_metric(image)
        if self.pixels_per_metric:
            print(f"Calibrated: {self.pixels_per_metric:.2f} pixels per cm (in warped space)")
        return self.pixels_per_metric is not None
    
    def measure_contour(self, contour):
        """
        Measure dimensions of a contour in real-world units.

        Steps:
        - contour is in ORIGINAL image coordinates.
        - warp contour into top-down metric space using homography H.
        - compute min-area rectangle in warped space.
        - convert width/height from pixels -> cm using pixels_per_metric.
        - project that rectangle's corners & center back to original for drawing.
        """
        if self.pixels_per_metric is None:
            raise ValueError("Must calibrate before measuring")
        
        if not hasattr(self.calibration, "H") or self.calibration.H is None:
            raise ValueError("Homography not available in calibration")
        
        # Ensure float32 and correct shape (Nx1x2)
        contour = contour.astype(np.float32)
        
        # Warp contour points into top-down metric space
        contour_warped = cv2.perspectiveTransform(contour, self.calibration.H)  # Nx1x2
        
        # Get bounding box in warped space
        rect = cv2.minAreaRect(contour_warped)
        box_warped = cv2.boxPoints(rect)  # 4x2
        
        # Dimensions in pixels (in warped metric space)
        width_px = rect[1][0]
        height_px = rect[1][1]
        
        # Convert to centimeters using known scale in warped space
        width_cm = width_px / self.pixels_per_metric
        height_cm = height_px / self.pixels_per_metric
        
        # Project box corners back to ORIGINAL image space for drawing
        box_original = cv2.perspectiveTransform(
            box_warped.reshape(-1, 1, 2),
            self.calibration.H_inv
        )
        box_original = np.intp(box_original.reshape(-1, 2))
        
        # Center point back to original
        center_warped = np.array([[rect[0]]], dtype=np.float32)  # 1x1x2
        center_original = cv2.perspectiveTransform(center_warped, self.calibration.H_inv)
        center_original = center_original[0][0]
        
        # Area in pixels (original contour)
        area_px = cv2.contourArea(contour)
        
        return {
            'width_cm': width_cm,
            'height_cm': height_cm,
            'box': box_original,             # in original coordinates
            'center': center_original,       # in original coordinates
            'area_px': area_px
        }


def main():
    cap = open_source(None)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Use homography calibration with a letter-size paper (8.5" x 11" = 21.6 x 27.9 cm)
    # Or use A4 paper: 21.0 x 29.7 cm
    print("\n=== Camera Calibration ===")
    print("Place a rectangular reference object (e.g., letter paper) flat on the surface.")
    print("Make sure all 4 corners are clearly visible.\n")
    
    calibration = HomographyCalibration(ref_width_cm=8.56, ref_height_cm=5.398)
    measurer = ObjectMeasurement(calibration)
    
    # Calibrate
    print("Capturing calibration frame...")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return
    
    if measurer.calibrate(frame):
        print("Calibration successful! Starting measurement stream...\n")
    else:
        print("Calibration failed")
        return
    
    # Create windows
    cv2.namedWindow('Measurements', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls')
    cv2.resizeWindow('Measurements', 1280, 720)
    
    # Trackbars for tuning detection
    cv2.createTrackbar('Canny Low', 'Controls', 50, 255, lambda x: None)
    cv2.createTrackbar('Canny High', 'Controls', 150, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 500, 10000, lambda x: None)
    cv2.createTrackbar('Show Edges', 'Controls', 0, 1, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_display = frame.copy()
        
        # Get parameters
        canny_low = cv2.getTrackbarPos('Canny Low', 'Controls')
        canny_high = cv2.getTrackbarPos('Canny High', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        show_edges = cv2.getTrackbarPos('Show Edges', 'Controls')
        
        # Build a mask to exclude the reference card region in ORIGINAL image
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        ref_points = calibration.get_reference_points(frame)
        if len(ref_points) == 4:
            ref_poly = np.array(ref_points, dtype=np.int32)
            cv2.fillPoly(mask, [ref_poly], 0)  # 0 = black = excluded
        
        # Detect objects in ORIGINAL image (excluding card region)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)  # Apply mask
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)  # Apply mask to edges too
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            try:
                # Measure by warping contour into metric space (homography) internally
                measurements = measurer.measure_contour(contour)
                detected += 1
                
                box_original = measurements['box']
                center_original = measurements['center']
                
                # Draw on ORIGINAL image
                cv2.drawContours(frame_display, [box_original], 0, (0, 255, 0), 2)
                
                # Measurement text
                text = f"{measurements['width_cm']:.1f} x {measurements['height_cm']:.1f} cm"
                text_pos = (int(center_original[0] - 70), int(center_original[1] - 10))
                
                # Black background for text
                cv2.rectangle(frame_display, 
                              (text_pos[0]-5, text_pos[1]-25), 
                              (text_pos[0]+200, text_pos[1]+10), 
                              (0, 0, 0), -1)
                
                cv2.putText(frame_display, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                # If something explodes for a weird contour, skip it
                # print("Measurement error:", e)
                continue
        
        # Display reference points on original view
        ref_points = calibration.get_reference_points(frame)
        for i, pt in enumerate(ref_points):
            cv2.circle(frame_display, pt, 8, (255, 0, 0), -1)
            cv2.putText(frame_display, str(i+1), (pt[0]+15, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw reference rectangle
        if len(ref_points) == 4:
            for i in range(4):
                cv2.line(frame_display, ref_points[i], ref_points[(i+1) % 4], (255, 0, 0), 2)
        
        # Info overlay
        cv2.rectangle(frame_display, (5, 5), (400, 90), (0, 0, 0), -1)
        info1 = f"Objects Detected: {detected}"
        info2 = f"Total Contours: {len(contours)}"
        info3 = "q=quit | r=recalibrate"
        
        cv2.putText(frame_display, info1, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame_display, info2, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame_display, info3, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Measurements', frame_display)
        
        # Show edge detection debug view
        if show_edges:
            cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Edge Detection', 640, 480)
            cv2.imshow('Edge Detection', edges)
        else:
            try:
                cv2.destroyWindow('Edge Detection')
            except:
                pass
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n=== Recalibrating ===")
            ret, frame = cap.read()
            if ret and measurer.calibrate(frame):
                print("Recalibration successful!\n")
            else:
                print("Recalibration failed\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
