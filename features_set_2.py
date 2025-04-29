import os
import cv2
import numpy as np
import shutil
from datetime import datetime

# Global masks
exclusion_masks = {}
center_masks = {}
visualize_steps = True

# -----------------------
# Helper: filter by centers
# -----------------------
import cv2
from typing import List, Tuple

def filter_contours_by_centers(
    contours: List[np.ndarray],
    centers: List[Tuple[int, int]],
    method: str = "sparse",
    threshold: float = 20.0
) -> List[np.ndarray]:
    """
    - 'sparse' mode: keep any contour whose signed distance from a center
      is ≥ -threshold (so contains the center or comes within threshold px outside).
    - 'thick' mode: keep any contour that actually contains the center (dist > 0).
    """
    kept = []
    for c in contours:
        for (x, y) in centers:
            dist = cv2.pointPolygonTest(c, (float(x), float(y)), True)
            if method == "sparse":
                if dist >= -threshold:
                    kept.append(c)
                    break
            else:  # thick
                if dist > 0:
                    kept.append(c)
                    break
    return kept

def find_centers_in_contours(
    contours: list[np.ndarray],
    centers_path: str,
    annotated: np.ndarray, method: str
) -> list[np.ndarray]:
    """
    Load centers from centers_path, filter contours to those containing a center,
    and—if visualize_steps—draw those filtered contours in dark green on annotated.
    Returns the filtered contour list.
    """
    if centers_path is None:
        return []
    centers = detect_nonwhite_clusters(centers_path)
    filtered = filter_contours_by_centers(contours, centers, method)
    if visualize_steps:
        # dark green in BGR:
        cv2.drawContours(annotated, filtered, -1, (0, 100, 0), 2)


    # Now loop thorugh the contours and if multiple centers are 
    return filtered

# -----------------------
# Colony-detection funcs
# -----------------------
def sparse_colonies(folder, image, base, exclusion_mask=None,
                    blocksize=51, C=2, min_area=30, max_area=2000):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        return [], 0, 0, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    allowed_mask = (1 - exclusion_mask) if exclusion_mask is not None else np.ones((h, w), np.uint8)

    # background subtraction
    bg = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    )
    corr = cv2.subtract(gray, bg)

    # threshold
    if '6A' in base:
        _, adapt = cv2.threshold(corr, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        adapt = cv2.adaptiveThreshold(
            corr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=blocksize,
            C=C
        )

    # mask exclusion regions
    adapt = cv2.bitwise_and(adapt, adapt, mask=allowed_mask * 255)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  k3, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k3, iterations=2)

    cnts, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, areas = [], []

    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        if exclusion_mask is not None and exclusion_mask[cy, cx] == 1:
            continue
        kept.append(c)
        areas.append(area)

    total_area = sum(areas)
    num_colonies = len(areas)

    if visualize_steps:
        out = img.copy()
        cv2.drawContours(out, kept, -1, (0, 255, 0), 2)
        return kept, total_area, num_colonies, out
    else:
        return kept, total_area, num_colonies, None

def thick_colonies(folder, image, base, exclusion_mask=None):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        return [], 0, 0, None

    h, w = img.shape[:2]
    allowed_mask = (1 - exclusion_mask) if exclusion_mask is not None else np.ones((h, w), np.uint8)
    masked = cv2.bitwise_and(img, img, mask=allowed_mask * 255)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 70, 60])
    upper_yellow = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k3, iterations=2)
    clean = cv2.dilate(clean, k3, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, areas = [], []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        if exclusion_mask is not None and exclusion_mask[cy, cx] == 1:
            continue
        kept.append(c)
        areas.append(area)

    total_area = sum(areas)
    num_colonies = len(areas)

    if visualize_steps:
        out = img.copy()
        cv2.drawContours(out, kept, -1, (0, 255, 0), 2)
        return kept, total_area, num_colonies, out
    else:
        return kept, total_area, num_colonies, None

# -----------------------
# Other utilities
# -----------------------
def load_exclusion_mask(mask_path):
    img = cv2.imread(mask_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0,70,50]), np.array([10,255,255])
    lower_red2, upper_red2 = np.array([160,70,50]), np.array([180,255,255])
    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return (cv2.bitwise_or(m1, m2) > 0).astype(np.uint8)

def detect_nonwhite_clusters(image_path, white_thresh=255, min_size=0):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    mask = np.any(img < white_thresh, axis=2).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    centers = []
    for lbl in range(1, labels.max()+1):
        ys, xs = np.where(labels == lbl)
        if xs.size < min_size:
            continue
        centers.append((int(xs.mean()), int(ys.mean())))
    return centers

def select_method(base, date):
    if base in {"0A", "0B"}:
        return "thick" if date >= "20250424_165952" else "sparse"
    if base == "3A":
        return "sparse"
    if base == "3B":
        return "thick" if date >= "20250427_055523" else "sparse"
    if base == "6A":
        return "thick"
    if base == "6B":
        return "thick" if date >= "20250426_085425" else "sparse"
    if base == "9A":
        return "thick" if date >= "20250427_085531" else "sparse"
    if base == "9B":
        if date < "20250425_182344":
            return "thick"
        elif date < "20250426_135438":
            return "sparse"
        else:
            return "thick"
    return "sparse"

earliest = datetime.strptime("20250422_164340", "%Y%m%d_%H%M%S")
def get_timesteps_since(date_str: str) -> float:
    dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    return (dt - earliest).total_seconds()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    base_dir = "cropped"
    if os.path.exists("centers"):
        shutil.rmtree("centers")

    plates = ["0A","0B","3A","3B","6A","6B","9A","9B"]
    for plate in plates:
        mp = os.path.join("border_masks", f"{plate}_mask.png")
        if os.path.exists(mp):
            exclusion_masks[plate] = load_exclusion_mask(mp)
        cp = os.path.join("center_masks", f"{plate}_centers.png")
        if os.path.exists(cp):
            center_masks[plate] = cp

    alpha = 0.5
    radius = 2  # circle radius for centers

    for date in sorted(os.listdir(base_dir)):
        date_dir = os.path.join(base_dir, date)
        if not os.path.isdir(date_dir):
            continue

        output_folder = os.path.join("centers", date)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(date_dir):
            if not filename.lower().endswith(('.jpg','.jpeg','.png')):
                continue

            base, _ = os.path.splitext(filename)
            mask = exclusion_masks.get(base)
            centers_path = center_masks.get(base)

            # detect colonies
            method = select_method(base, date)
            if method == "thick":
                kept_contours, total_area, num_colonies, annotated = thick_colonies(
                    date_dir, filename, base, exclusion_mask=mask
                )
            else:
                kept_contours, total_area, num_colonies, annotated = sparse_colonies(
                    date_dir, filename, base, exclusion_mask=mask
                )

            # filter by center containment & plot in dark green if requested
            if annotated is not None:
                kept_contours = find_centers_in_contours(
                    kept_contours,
                    centers_path,
                    annotated, method
                )

                # then the existing red + blue overlays …
                h, w = annotated.shape[:2]
                if mask is not None:
                    m = mask.astype(bool)
                    # red overlay
                    annotated[...,2][m] = (
                        (1-alpha)*annotated[...,2][m] + alpha*255
                    ).astype(np.uint8)
                    annotated[...,0][m] = (annotated[...,0][m] * (1-alpha)).astype(np.uint8)
                    annotated[...,1][m] = (annotated[...,1][m] * (1-alpha)).astype(np.uint8)

                if centers_path:
                    centers = detect_nonwhite_clusters(centers_path)
                    for x, y in centers:
                        x0, x1 = max(x-radius, 0), min(x+radius+1, w)
                        y0, y1 = max(y-radius, 0), min(y+radius+1, h)
                        roi = annotated[y0:y1, x0:x1]
                        yy, xx = np.ogrid[y0:y1, x0:x1]
                        circle_mask = ((xx - x)**2 + (yy - y)**2) <= radius**2

                        # blue circle
                        roi[...,0][circle_mask] = (
                            (1-alpha)*roi[...,0][circle_mask] + alpha*255
                        ).astype(np.uint8)
                        roi[...,1][circle_mask] = (roi[...,1][circle_mask] * (1-alpha)).astype(np.uint8)
                        roi[...,2][circle_mask] = (roi[...,2][circle_mask] * (1-alpha)).astype(np.uint8)
                        annotated[y0:y1, x0:x1] = roi

                out_path = os.path.join(output_folder, f"{base}.png")
                cv2.imwrite(out_path, annotated)
                print(f"Saved overlay for {base} → {out_path}")
