import os
import cv2
import numpy as np
from final_project import load_exclusion_mask
import shutil
from datetime import datetime

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


# UPDATED: detect red-point centers on white background
# get centers from an image with red pixels on white background
def detect_nonwhite_clusters(image_path, white_thresh=255, min_size=0):
    """
    Detect one representative coordinate per cluster of non-white pixels.

    Args:
        image_path (str): path to input image
        white_thresh (int): anything with all channels >= this is considered white
        min_size (int): drop clusters smaller than this many pixels

    Returns:
        List[Tuple[int,int]]: list of (x,y) cluster centroids
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # 1. build a binary mask where any pixel not “almost white” is 1
    #    i.e. if ALL channels >= white_thresh → white → mask=0
    mask = np.any(img < white_thresh, axis=2).astype(np.uint8)

    # 2. optional: remove tiny specks with opening
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. connected components
    num_labels, labels = cv2.connectedComponents(mask)

    centers = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if xs.size < min_size:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        centers.append((cx, cy))

    return centers

EARLIEST = datetime.strptime("20250422_164340", "%Y%m%d_%H%M%S")
def get_timesteps_since(date_str: str) -> float:
    fmt = "%Y%m%d_%H%M%S"
    dt = datetime.strptime(date_str, fmt)
    delta = dt - EARLIEST
    return delta.total_seconds()


if __name__ == "__main__":
    # load exclusion mask for plate 6A
    mask = load_exclusion_mask('6A_mask.png')

    # run the thick-colonies pipeline on 6A.jpg
    contours, total_area, num_colonies, annotated_image = thick_colonies(
        'cropped/20250428_095641',
        '6A.jpg',
        '6A',
        exclusion_mask=mask
    )

    # detect the red-point centers from your centers image
    centers_img_path = '6A_centers.png'
    colony_centers = detect_nonwhite_clusters(centers_img_path)
    print("Detected centers:", colony_centers)

    # overlay those centers directly onto annotated_image
    for (x, y) in colony_centers:
        cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)


    # 1. Load the raw centers‐image
    overlay = cv2.imread(centers_img_path)
    if overlay is None:
        raise FileNotFoundError(f"Cannot load overlay: {centers_img_path}")

    # 2. Resize overlay to match annotated_image, if needed
    h, w = annotated_image.shape[:2]
    if (overlay.shape[0], overlay.shape[1]) != (h, w):
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # 3. Blend at 50% opacity
    alpha = 0.5
    blended = cv2.addWeighted(annotated_image, alpha,
                              overlay,          1 - alpha,
                              0)



    # 5. Show the final result
    cv2.imshow('Annotated + Centers Overlay', blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show result
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
