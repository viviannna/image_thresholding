import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.stats import norm
import pickle as pk

def segment_growth_strict_color(folder, image):
    img_path = os.path.join(folder, image)
    img = cv.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 10

    # Circular mask
    Y, X = np.ogrid[:h, :w]
    mask = ((X - center[0])**2 + (Y - center[1])**2) <= radius**2

    # Mask image
    masked_img = np.zeros_like(img)
    masked_img[mask] = img[mask]

    # Convert to HSV
    hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)

    # 👇 MUCH tighter yellow hue, and higher minimum saturation
    lower_yellow = np.array([20, 70, 60])   # was too low before
    upper_yellow = np.array([35, 255, 255])
    color_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cleaned = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv.dilate(cleaned, kernel, iterations=1)

    # Remove small blobs
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(cleaned)
    min_area = 200  # ← increased again
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 255

    # Compute area
    growth_area = np.sum(final_mask > 0)
    plate_area = np.sum(mask)
    percent_covered = 100 * growth_area / plate_area

    # Draw overlay
    result = img.copy()
    result[final_mask > 0] = [0, 255, 0]
    cv.circle(result, center, radius, (255, 0, 0), 2)

    os.makedirs(os.path.join(folder, "growth_strict_color"), exist_ok=True)
    out_path = os.path.join(folder, "growth_strict_color", image)
    cv.imwrite(out_path, result)

    print(f"[✓] Saved to {out_path}")
    print(f"Growth area: {growth_area} pixels")
    print(f"Percent of plate covered: {percent_covered:.2f}%")

    return growth_area, percent_covered


def extract_features(contours):
    """
    Given the contours of an image, extract the features of interest.
    """

    contours = [c for c in contours if cv2.contourArea(c) > 0]
    num_cells = len(contours)

    total_area = 0.0
    total_perimeter = 0.0
    total_area = 0.0
    total_ratio = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        total_area += area
        total_perimeter += perimeter
        total_area += area

    average_area = total_area / num_cells if num_cells else 0
    average_perimeter = total_perimeter / num_cells if num_cells else 0


    feature_array = np.array([
        average_area,
        average_perimeter,
        total_area,
        total_perimeter,
        num_cells
    ])

    return feature_array



if __name__ == "__main__":
    
    image_folder = "20250425_083100"
    for image_file in os.listdir(image_folder):
        segment_growth_strict_color(image_folder, image_file)

    image_folder = "20250423_205819"
    for image_file in os.listdir(image_folder):
        segment_growth_strict_color(image_folder, image_file)
    