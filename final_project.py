import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier      # you already have these
# … your other imports …

def thick_colonies(folder, image):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)


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
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    # MUCH tighter yellow hue, and higher minimum saturation
    lower_yellow = np.array([20, 70, 60])
    upper_yellow = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    min_area = 200
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 255

    # Compute area (if you still need these numbers)
    growth_area = np.sum(final_mask > 0)
    plate_area  = np.sum(mask)
    percent_covered = 100 * growth_area / plate_area

    # Draw overlay
    result = img.copy()
    result[final_mask > 0] = [0, 255, 0]
    cv2.circle(result, center, radius, (255, 0, 0), 2)

    # ─── NEW SAVING LOGIC ───────────────────────────────────
    base, ext = os.path.splitext(image)      # "0A", ".jpg"
    out_dir = os.path.join(folder, base)     # e.g. "20250422_164340/0A"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"thick{ext}")
    cv2.imwrite(out_path, result)
    # print(f"[✓] thick → {out_path}")
    # ────────────────────────────────────────────────────────

    return growth_area, percent_covered


def sparse_colonies(folder, image,
                    rim_border=10,
                    blocksize=51,
                    C=2,
                    min_area=30,
                    max_area=2000):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plate detection (Hough)
    blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=h/4,
                               param1=50, param2=30,
                               minRadius=int(min(w,h)*0.3),
                               maxRadius=int(min(w,h)*0.6))
    if circles is not None:
        x,y,r = circles[0,0]
        center, radius = (int(x),int(y)), int(r)
    else:
        center = (w//2, h//2)
        radius = min(center) - 10

    # mask just inside real rim
    effective_r = radius - rim_border
    mask_full = np.zeros((h, w), np.uint8)
    cv2.circle(mask_full, center, effective_r, 255, -1)

    # debug overlay
    out = img.copy()
    cv2.circle(out, center, radius,      (255,0,0), 2)  # true rim = blue
    cv2.circle(out, center, effective_r, (0,0,255), 1)  # mask edge = red

    # background flatten + adaptive inverted threshold
    bg   = cv2.morphologyEx(
             gray, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
           )
    corr = cv2.subtract(gray, bg)
    adapt = cv2.adaptiveThreshold(
              corr, 255,
              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
              cv2.THRESH_BINARY_INV,
              blockSize=blocksize,
              C=C
            )
    adapt = cv2.bitwise_and(adapt, adapt, mask=mask_full)

    # clean small noise & fill
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  k3, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k3, iterations=2)

    # find & rim‐filter contours
    cnts, _ = cv2.findContours(adapt,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    colonies = []
    for c in cnts:
        A = cv2.contourArea(c)
        if not (min_area <= A <= max_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        dist = np.hypot(cx-center[0], cy-center[1])
        if dist > (effective_r - rim_border):
            continue
        colonies.append(c)

    # draw
    cv2.drawContours(out, colonies, -1, (0,255,0), 2)

    # ─── NEW SAVING LOGIC ───────────────────────────────────
    base, ext = os.path.splitext(image)      # "0A", ".jpg"
    out_dir = os.path.join(folder, base)     # e.g. "20250422_164340/0A"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"thin{ext}")
    cv2.imwrite(out_path, out)
    # print(f"[✓] thin  → {out_path}")
    # ────────────────────────────────────────────────────────

    return out

if __name__ == "__main__":
    base_dir = "cropped"
    for date in os.listdir(base_dir):
        date_dir = os.path.join(base_dir, date)
        if not os.path.isdir(date_dir): 
            continue
        for img in os.listdir(date_dir):
            if not img.lower().endswith((".jpg","jpeg","png")):
                continue

            # for each cropped/date/img.jpg, these two calls will
            # create cropped/date/img/ thick.jpg and thin.jpg

            #

            
            thick_colonies(date_dir, img)
            sparse_colonies(date_dir, img)
