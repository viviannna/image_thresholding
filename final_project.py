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

# def segment_growth_strict_color(folder, image):
#     img_path = os.path.join(folder, image)
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Could not load: {img_path}")
#         return

#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     radius = min(center) - 10

#     # Circular mask
#     Y, X = np.ogrid[:h, :w]
#     mask = ((X - center[0])**2 + (Y - center[1])**2) <= radius**2

#     # Mask image
#     masked_img = np.zeros_like(img)
#     masked_img[mask] = img[mask]

#     # Convert to HSV
#     hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

#     # 👇 MUCH tighter yellow hue, and higher minimum saturation
#     lower_yellow = np.array([20, 70, 60])   # was too low before
#     upper_yellow = np.array([35, 255, 255])
#     color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     # Clean up
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
#     cleaned = cv2.dilate(cleaned, kernel, iterations=1)

#     # Remove small blobs
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
#     min_area = 200  # ← increased again
#     final_mask = np.zeros_like(cleaned)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             final_mask[labels == i] = 255

#     # Compute area
#     growth_area = np.sum(final_mask > 0)
#     plate_area = np.sum(mask)
#     percent_covered = 100 * growth_area / plate_area

#     # Draw overlay
#     result = img.copy()
#     result[final_mask > 0] = [0, 255, 0]
#     cv2.circle(result, center, radius, (255, 0, 0), 2)

#     # os.makedirs(os.path.join(folder, "growth_strict_color"), exist_ok=True)
   
#     # out_path = os.path.join(folder, "growth_strict_color", image)
#     # cv2.imwrite(out_path, result)

#     out_dir = os.path.join("growth_strict_color", folder)
#     os.makedirs(out_dir, exist_ok=True)  # Create the folders if they don't exist

#     out_path = os.path.join(out_dir, image)
#     cv2.imwrite(out_path, result)

#     print(f"[✓] Saved to {out_path}")
#     print(f"Growth area: {growth_area} pixels")
#     print(f"Percent of plate covered: {percent_covered:.2f}%")

#     return growth_area, percent_covered
fn_name = "threshold_growth_hough"

fn_name = "threshold_growth_colonies"

def threshold_growth_colonies(folder,
                              image,
                              shrink_frac=0.80,
                              blocksize=51,
                              C=2,
                              min_area=50):
    """
    1) Auto-detect plate via HoughCircles
    2) Mask only inside a smaller circle (shrink_frac*radius)
    3) Flatten lighting, then adaptive inverted threshold (to pick dark colonies)
    4) Morph-open to clean noise
    5) Filter contours by area (min_area)
    6) Draw colonies in red
    """
    # 1) load + gray
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) detect plate circle
    blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=h/4, param1=50, param2=30,
                               minRadius=int(min(w, h)*0.3),
                               maxRadius=int(min(w, h)*0.6))
    if circles is not None:
        x, y, r = circles[0, 0]
        center, radius = (int(x), int(y)), int(r)
    else:
        center = (w//2, h//2)
        radius = min(center) - 10

    # 3) build smaller mask & draw true plate rim
    small_r = int(radius * shrink_frac)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, small_r, 255, -1)
    output = img.copy()
    cv2.circle(output, center, radius, (255, 0, 0), 2)       # blue true rim
    cv2.circle(output, center, small_r, (0, 0, 255), 1)      # red inner mask

    # 4) flatten background
    inside = cv2.bitwise_and(gray, gray, mask=mask)
    bg = cv2.morphologyEx(inside,
                          cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)))
    corr = cv2.subtract(inside, bg)

    # 5) adaptive inverted threshold → dark colonies = white
    adapt = cv2.adaptiveThreshold(corr, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  blockSize=blocksize,
                                  C=C)

    # 6) clean small noise
    clean = cv2.morphologyEx(adapt,
                             cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                             iterations=2)

    # 7) find/filter contours
    cnts, _ = cv2.findContours(clean,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    colonies = [c for c in cnts if cv2.contourArea(c) >= min_area]

    # 8) draw
    cv2.drawContours(output, colonies, -1, (0, 0, 255), 2)   # red

    # 9) save
    out_dir = os.path.join(fn_name, folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, image)
    cv2.imwrite(out_path, output)
    print(f"Saved to {out_path}  ({len(colonies)} colonies)")

    return output


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
        threshold_growth_colonies(image_folder, image_file)

    image_folder = "20250423_205819"
    for image_file in os.listdir(image_folder):
        threshold_growth_colonies(image_folder, image_file)
    