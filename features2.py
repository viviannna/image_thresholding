import os
import cv2
import numpy as np
import shutil
from datetime import datetime

# Global masks
exclusion_masks = {}
center_masks = {}

# Sparse-colony detection with exclusion mask applied
def sparse_colonies(folder, image, base, exclusion_mask=None,
                    blocksize=51, C=2, min_area=30, max_area=2000):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        return [], 0, 0, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply exclusion mask if provided
    allowed_mask = (1 - exclusion_mask) if exclusion_mask is not None else np.ones((h, w), np.uint8)

    # Background subtraction
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)))
    corr = cv2.subtract(gray, bg)

    # Adaptive threshold or simple for 6A
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

    # Mask exclusion regions
    adapt = cv2.bitwise_and(adapt, adapt, mask=allowed_mask * 255)

    # Clean up mask
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  k3, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k3, iterations=2)

    cnts, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, areas = [], []

    # Filter and remove excluded-centroid contours
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if exclusion_mask is not None and exclusion_mask[cy, cx] == 1:
            continue
        kept.append(c)
        areas.append(area)

    total_area = sum(areas)
    num_colonies = len(areas)

    out = img.copy()
    cv2.drawContours(out, kept, -1, (0, 255, 0), 2)
    return kept, total_area, num_colonies, out

# Thick-colony detection with exclusion mask
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
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if exclusion_mask is not None and exclusion_mask[cy, cx] == 1:
            continue
        kept.append(c)
        areas.append(area)

    total_area = sum(areas)
    num_colonies = len(areas)

    out = img.copy()
    cv2.drawContours(out, kept, -1, (0, 255, 0), 2)
    return kept, total_area, num_colonies, out

# Load a red-based exclusion mask
def load_exclusion_mask(mask_path):
    img = cv2.imread(mask_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    full_red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    return (full_red_mask > 0).astype(np.uint8)

# Calculate seconds since earliest timestamp
earliest = datetime.strptime("20250422_164340", "%Y%m%d_%H%M%S")
def get_timesteps_since(date_str: str) -> float:
    dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    return (dt - earliest).total_seconds()

# Logic to pick detection method per plate
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

# Detect non-white clusters for centers
def detect_nonwhite_clusters(image_path, white_thresh=255, min_size=0):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    mask = np.any(img < white_thresh, axis=2).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)

    centers = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if xs.size < min_size:
            continue
        centers.append((int(xs.mean()), int(ys.mean())))
    return centers

if __name__ == "__main__":
    base_dir = "cropped"

    # Clear old outputs
    if os.path.exists("centers"):
        shutil.rmtree("centers")

    plates = ["0A","0B","3A","3B","6A","6B","9A","9B"]

    # Load exclusion masks from border_masks/
    for plate in plates:
        mp = os.path.join("border_masks", f"{plate}_mask.png")
        if os.path.exists(mp):
            exclusion_masks[plate] = load_exclusion_mask(mp)

    # Load center masks from center_masks/
    for plate in plates:
        cp = os.path.join("center_masks", f"{plate}_centers.png")
        if os.path.exists(cp):
            center_masks[plate] = load_exclusion_mask(cp)

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
            center_mask = center_masks.get(base)

            # Run detection pipeline
            method = select_method(base, date)
            if method == "thick":
                contours, total_area, num_colonies, annotated = thick_colonies(
                    date_dir, filename, base, exclusion_mask=mask
                )
            else:
                contours, total_area, num_colonies, annotated = sparse_colonies(
                    date_dir, filename, base, exclusion_mask=mask
                )

            # Blend in the raw-centers image
            centers_path = os.path.join("center_masks", f"{base}_centers.png")
            overlay = cv2.imread(centers_path)
            if overlay is None:
                raise FileNotFoundError(f"Cannot load overlay: {centers_path}")
            h, w = annotated.shape[:2]
            if overlay.shape[:2] != (h, w):
                overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
            blended = cv2.addWeighted(annotated, 0.5, overlay, 0.5, 0)

            # Overlay shaded exclusion mask in semi-transparent red
            if mask is not None:
                red_layer = np.zeros_like(blended)
                red_layer[:, :, 2] = 255  # full red

                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_3c = cv2.merge([mask_uint8]*3)

                red_masked = cv2.bitwise_and(red_layer, mask_3c)
                blended = cv2.addWeighted(blended, 1.0, red_masked, 0.5, 0)

            # Plot center coordinates with semi-transparent blue markers
            if center_mask is not None:
                pts = detect_nonwhite_clusters(os.path.join("center_masks", f"{base}_centers.png"))
                overlay_pts = blended.copy()
                for (x, y) in pts:
                    cv2.circle(overlay_pts, (x, y), 5, (255, 0, 0), -1)  # blue BGR
                blended = cv2.addWeighted(overlay_pts, 0.5, blended, 0.5, 0)

            # Save the final overlay
            out_path = os.path.join(output_folder, f"{base}.png")
            cv2.imwrite(out_path, blended)
            print(f"Saved overlay for {base} → {out_path}")
