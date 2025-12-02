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

        # Store warped dimensions of the reference
        self.warp_width_px = None
        self.warp_height_px = None
        
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
            self.warp_width_px = width_px
            self.warp_height_px = height_px
            
            dst_pts = np.float32([
                [0, 0],
                [width_px, 0],
                [width_px, height_px],
                [0, height_px]
            ])
            
            # Compute homography: original -> top-down metric space
            self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            print("Homography computed successfully!")

            # Debug: measure the reference itself in warped space
            card_cnt = src_pts.reshape(-1, 1, 2).astype(np.float32)
            card_warped = cv2.perspectiveTransform(card_cnt, self.H)  # 4x1x2
            rect = cv2.minAreaRect(card_warped.reshape(-1, 2))
            w_px, h_px = rect[1]
            w_cm_est = w_px / self.pixels_per_cm_warped
            h_cm_est = h_px / self.pixels_per_cm_warped
            print(f"[Debug] Reference in warped space ≈ {w_cm_est:.3f} x {h_cm_est:.3f} cm "
                  f"(target {self.ref_width_cm:.3f} x {self.ref_height_cm:.3f} cm)")
            
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

    def draw_grid(self, image, grid_step_cm=1.0, extent_factor=3.0):
        """
        Draw a perspective-correct metric grid on the ORIGINAL image.

        grid_step_cm: spacing between grid lines in cm.
        extent_factor: how far the grid extends relative to the card size.
        """
        if self.H is None or self.H_inv is None:
            return image

        ppc = self.pixels_per_cm_warped
        if self.warp_width_px is None or self.warp_height_px is None:
            warp_w = self.ref_width_cm * ppc
            warp_h = self.ref_height_cm * ppc
        else:
            warp_w = self.warp_width_px
            warp_h = self.warp_height_px

        half_w = warp_w * extent_factor
        half_h = warp_h * extent_factor

        step_px = grid_step_cm * ppc
        if step_px <= 0:
            return image

        img_out = image.copy()

        # Vertical lines (constant x in warped space)
        xs = np.arange(-half_w, half_w + step_px, step_px)
        for x in xs:
            p1 = np.array([[[x, -half_h]]], dtype=np.float32)  # (1,1,2)
            p2 = np.array([[[x,  half_h]]], dtype=np.float32)
            pts = np.concatenate([p1, p2], axis=0)             # (2,1,2)

            pts_img = cv2.perspectiveTransform(pts, self.H_inv)  # (2,1,2) in original
            pts_img = pts_img.reshape(-1, 2)
            p1_img = tuple(np.intp(pts_img[0]))
            p2_img = tuple(np.intp(pts_img[1]))

            cv2.line(img_out, p1_img, p2_img, (80, 80, 80), 1, lineType=cv2.LINE_AA)

        # Horizontal lines (constant y in warped space)
        ys = np.arange(-half_h, half_h + step_px, step_px)
        for y in ys:
            p1 = np.array([[[-half_w, y]]], dtype=np.float32)
            p2 = np.array([[[ half_w, y]]], dtype=np.float32)
            pts = np.concatenate([p1, p2], axis=0)

            pts_img = cv2.perspectiveTransform(pts, self.H_inv)
            pts_img = pts_img.reshape(-1, 2)
            p1_img = tuple(np.intp(pts_img[0]))
            p2_img = tuple(np.intp(pts_img[1]))

            cv2.line(img_out, p1_img, p2_img, (80, 80, 80), 1, lineType=cv2.LINE_AA)

        # Optional: show a 10 cm indicator near the card
        if len(self.points) == 4:
            tl = np.array(self.points[0], dtype=np.float32)
            tl_warp = np.array([[tl]], dtype=np.float32)  # 1x1x2
            tl_metric = cv2.perspectiveTransform(tl_warp, self.H)  # to warped space

            # A point 10 cm to the right in metric space
            p2_metric = tl_metric.copy()
            p2_metric[0, 0, 0] += 10 * ppc  # +10 cm in x

            # Back to original
            two_pts = np.concatenate([tl_metric, p2_metric], axis=0)  # 2x1x2
            two_img = cv2.perspectiveTransform(two_pts, self.H_inv).reshape(-1, 2)
            p1_img = tuple(np.intp(two_img[0]))
            p2_img = tuple(np.intp(two_img[1]))

            cv2.line(img_out, p1_img, p2_img, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(img_out, "10cm", (p2_img[0] + 5, p2_img[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return img_out


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

        contour: in ORIGINAL image coordinates (as from findContours).
        """
        if self.pixels_per_metric is None:
            raise ValueError("Must calibrate before measuring")
        
        if not hasattr(self.calibration, "H") or self.calibration.H is None:
            raise ValueError("Homography not available in calibration")
        
        # Ensure float32 and proper shape
        contour = contour.astype(np.float32)  # (N, 1, 2)
        
        # Warp contour points into top-down metric space
        contour_warped = cv2.perspectiveTransform(contour, self.calibration.H)  # (N, 1, 2)
        cnt_w = contour_warped.reshape(-1, 2)  # (N, 2)
        
        if cnt_w.shape[0] < 3:
            raise ValueError("Contour too small for measurement")

        # Get bounding box in warped space
        rect = cv2.minAreaRect(cnt_w)
        box_warped = cv2.boxPoints(rect)  # 4x2
        
        # Dimensions in pixels (warped metric space)
        width_px = rect[1][0]
        height_px = rect[1][1]
        
        # Convert to centimeters using known scale in warped space
        width_cm = width_px / self.pixels_per_metric
        height_cm = height_px / self.pixels_per_metric
        
        # Project box corners back to ORIGINAL image space for drawing
        box_original = cv2.perspectiveTransform(
            box_warped.reshape(-1, 1, 2).astype(np.float32),
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
    
    # Use homography calibration with a credit card (approx 8.56 x 5.398 cm)
    print("\n=== Camera Calibration ===")
    print("Place a credit card (or card-sized rectangle) flat on the surface.")
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
    
    # Trackbars for tuning detection & grid
    cv2.createTrackbar('Canny Low', 'Controls', 50, 255, lambda x: None)
    cv2.createTrackbar('Canny High', 'Controls', 150, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Controls', 500, 100000, lambda x: None)
    cv2.createTrackbar('Show Edges', 'Controls', 0, 1, lambda x: None)
    cv2.createTrackbar('Grid Step cm', 'Controls', 2, 20, lambda x: None)
    
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
        grid_step_cm = cv2.getTrackbarPos('Grid Step cm', 'Controls')
        if grid_step_cm <= 0:
            grid_step_cm = 1
        
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
                
            except Exception:
                continue
        
        # Draw metric grid on top of the original frame
        frame_display = calibration.draw_grid(
            frame_display,
            grid_step_cm=float(grid_step_cm),
            extent_factor=4.0
        )
        
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
        cv2.rectangle(frame_display, (5, 5), (430, 100), (0, 0, 0), -1)
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