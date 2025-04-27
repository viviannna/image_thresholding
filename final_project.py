import os
import cv2
import numpy as np

# default amount to shrink the detected plate radius
PLATE_SHRINK = 20

def thick_colonies(folder, image):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return

    base = os.path.splitext(image)[0]

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 10

    # per-image shrink override
    shrink = PLATE_SHRINK
    if '0A' in base or '0B' in base:
        shrink = 0
    inner_r = radius - shrink

    # mask to inner circle
    Y, X = np.ogrid[:h, :w]
    mask = ((X - center[0])**2 + (Y - center[1])**2) <= inner_r**2

    # only keep pixels within that inner circle
    masked = np.zeros_like(img)
    masked[mask] = img[mask]

    # HSV threshold for thick (yellow) colonies
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 70, 60])
    upper_yellow = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # clean up small blobs
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k3, iterations=2)
    clean = cv2.dilate(clean, k3, iterations=1)

    # filter by size
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    final_mask = np.zeros_like(clean)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            final_mask[labels == i] = 255

    # draw results
    out = img.copy()
    out[final_mask > 0] = (0, 255, 0)
    cv2.circle(out, center, inner_r, (255, 0, 0), 2)

    # save
    date = os.path.basename(folder)
    base_name, ext = os.path.splitext(image)
    odir = os.path.join("thresholded", date)
    os.makedirs(odir, exist_ok=True)
    out_path = os.path.join(odir, base_name + ext)
    cv2.imwrite(out_path, out)
    print(f"[thick] → {out_path}")


def sparse_colonies(folder, image,
                    rim_border=10,
                    blocksize=51,
                    C=2,
                    min_area=30,
                    max_area=2000):
    img_path = os.path.join(folder, image)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return

    base = os.path.splitext(image)[0]
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- plate detection override for 6A ----
    if '6A' in base:
        # force-center + fallback radius
        center = (w//2, h//2)
        radius = min(center) - 10
    else:
        # standard HoughCircles detection
        blur = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=h/4,
            param1=50, param2=30,
            minRadius=int(min(w, h)*0.3),
            maxRadius=int(min(w, h)*0.6)
        )
        if circles is not None:
            x, y, r = circles[0, 0]
            center, radius = (int(x), int(y)), int(r)
        else:
            center = (w//2, h//2)
            radius = min(center) - 10

    # compute effective inner radius
    shrink = PLATE_SHRINK
    if '0A' in base or '0B' in base:
        shrink = 0
    eff_r = radius - rim_border - shrink

    # build circular mask
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, eff_r, 255, -1)

    # flatten background
    bg = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    )
    corr = cv2.subtract(gray, bg)

    # ---- thresholding: fixed for 6A, adaptive for others ----
    if '6A' in base:
        # PICK A VALUE here that works for your 6A lighting (e.g. 50)
        _, adapt = cv2.threshold(corr, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        adapt = cv2.adaptiveThreshold(
            corr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=blocksize,
            C=C
        )

    # apply circle mask
    adapt = cv2.bitwise_and(adapt, adapt, mask=mask)

    # clean up noise
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  k3, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k3, iterations=2)

    # find + filter contours
    cnts, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in cnts:
        A = cv2.contourArea(c)
        if not (min_area <= A <= max_area):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        if np.hypot(cx - center[0], cy - center[1]) > (eff_r - rim_border):
            continue
        kept.append(c)

    # draw result + inner circle
    out = img.copy()
    cv2.drawContours(out, kept, -1, (0, 255, 0), 2)
    cv2.circle(out, center, eff_r, (255, 0, 0), 2)

    # save
    date = os.path.basename(folder)
    base_name, ext = os.path.splitext(image)
    odir = os.path.join("thresholded", date)
    os.makedirs(odir, exist_ok=True)
    out_path = os.path.join(odir, base_name + ext)
    cv2.imwrite(out_path, out)
    print(f"[sparse] → {out_path}")


if __name__ == "__main__":
    base_dir = "cropped"
    cutoff = "20250424_165952"  # strictly before → sparse; after → thick

    for date in sorted(os.listdir(base_dir)):
        date_dir = os.path.join(base_dir, date)
        if not os.path.isdir(date_dir):
            continue

        for img in os.listdir(date_dir):
            if not img.lower().endswith((".jpg", "jpeg", "png")):
                continue

            base = os.path.splitext(img)[0]
            if base in {"3A","3B","6A","6B"}:
                sparse_colonies(date_dir, img)
            elif base in {"9A","9B"}:
                thick_colonies(date_dir, img)
            elif date < cutoff:
                sparse_colonies(date_dir, img)
            else:
                thick_colonies(date_dir, img)
