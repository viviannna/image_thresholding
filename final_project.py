import os
import cv2
import numpy as np
import shutil
from datetime import datetime
import csv
import pickle as pk

# Global
exclusion_masks = {}

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

def thick_colonies(folder, image, base, exclusion_mask=None):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0, None

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
    kept = []
    areas = []

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

    return total_area, num_colonies, out

def sparse_colonies(folder, image, base, exclusion_mask=None,
                    blocksize=51, C=2, min_area=30, max_area=2000):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    allowed_mask = (1 - exclusion_mask) if exclusion_mask is not None else np.ones((h, w), np.uint8)

    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)))
    corr = cv2.subtract(gray, bg)

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

    adapt = cv2.bitwise_and(adapt, adapt, mask=allowed_mask * 255)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  k3, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k3, iterations=2)

    cnts, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    areas = []

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

    return total_area, num_colonies, out

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
        elif "20250425_182344" <= date < "20250426_135438":
            return "sparse"
        else:
            return "thick"
    return "sparse"

EARLIEST = datetime.strptime("20250422_164340", "%Y%m%d_%H%M%S")

def get_timesteps_since(date_str: str) -> float:
    fmt = "%Y%m%d_%H%M%S"
    dt = datetime.strptime(date_str, fmt)
    delta = dt - EARLIEST
    return delta.total_seconds()

if __name__ == "__main__":

    if not os.path.exists('featureset1.pk'):
        base_dir = "cropped"

        # Delete thresholded/ if exists
        if os.path.exists("thresholded"):
            shutil.rmtree("thresholded")

        # Load exclusion masks
        for plate in ["0A", "0B", "3A", "3B", "6A", "6B", "9A", "9B"]:
            mask_path = f"{plate}_mask.png"
            if os.path.exists(mask_path):
                exclusion_masks[plate] = load_exclusion_mask(mask_path)

        feature_dict = {}

        for date in sorted(os.listdir(base_dir)):
            date_dir = os.path.join(base_dir, date)
            if not os.path.isdir(date_dir):
                continue

            timestep = get_timesteps_since(date)

            for img in os.listdir(date_dir):
                if not img.lower().endswith((".jpg", "jpeg", "png")):
                    continue

                base = os.path.splitext(img)[0]
                group = base[1]
                num_sprays = int(base[0])

                method = select_method(base, date)
                mask = exclusion_masks.get(base, None)

                if method == "thick":
                    area, num_colonies, out_img = thick_colonies(date_dir, img, base, exclusion_mask=mask)
                else:
                    area, num_colonies, out_img = sparse_colonies(date_dir, img, base, exclusion_mask=mask)

                # >>>> Overlay red mask onto output <<<<
                if out_img is not None and mask is not None:
                    overlay = out_img.copy()
                    red_overlay = np.zeros_like(out_img)
                    red_overlay[:, :, 2] = 255  # Red color
                    mask_3ch = np.stack([mask * 255] * 3, axis=-1)
                    overlay = np.where(mask_3ch == 255, (0.7 * overlay + 0.3 * red_overlay).astype(np.uint8), overlay)
                    out_img = overlay

                save_dir = os.path.join("thresholded", date)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, img), out_img)

                feature_tuple = (timestep, group, num_sprays)
                avg_area = area / num_colonies if num_colonies > 0 else 0
                feature_dict[feature_tuple] = (area, num_colonies, avg_area)

            with open('featureset1.pk', 'wb') as f:
                pk.dump(feature_dict, f)
            
    else:
        feature_dict = pk.load(open('featureset1.pk', 'rb'))

    data = []
    data.append(["time", "group", "numsprays", "area", "numcolonies", "avgarea"])
    for (key_tuple, image_features) in list(feature_dict.items()):
        line = list(key_tuple) + list(image_features)
        data.append(line)

    # Writing data to a CSV file
    with open('featureset1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("CSV file 'featureset1.csv' created successfully.")
