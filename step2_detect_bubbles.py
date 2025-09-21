import cv2
import numpy as np

# Paths
input_path = r"C:\Users\v-jananik\Downloads\OMR_Project\input\sample1.jpeg"
threshold_path = r"C:\Users\v-jananik\Downloads\OMR_Project\output\thresholded.jpg"
output_path = r"C:\Users\v-jananik\Downloads\OMR_Project\output\bubbles_detected.jpg"

# ------------------ Step 1: Preprocessing ------------------
image = cv2.imread(input_path)
if image is None:
    print("❌ Image not found!")
    exit()

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian blur
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35, 10
)

# Optional morphological closing to fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Save thresholded image
cv2.imwrite(threshold_path, thresh)
print("✅ Thresholded image saved:", threshold_path)

# ------------------ Step 2: Detect bubbles ------------------
# Invert for HoughCircles
inverted = cv2.bitwise_not(thresh)

# Hough Circle detection
circles = cv2.HoughCircles(
    inverted,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=22,
    minRadius=1,  # tune based on real bubble size
    maxRadius=10
)

output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
bubble_count = 0
filled_count = 0

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, (x, y, r) in enumerate(circles[0, :]):
        # Check if bubble is filled
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r-2, 255, -1)  # slightly smaller circle
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        circle_area = np.pi * (r-2)**2
        fill_ratio = filled_pixels / circle_area

        # Decide if bubble is filled (tune threshold 0.3~0.5)
        if fill_ratio > 0.35:
            color = (0, 0, 255)  # red for filled
            filled_count += 1
        else:
            color = (0, 255, 0)  # green for unfilled

        # Draw circle
        cv2.circle(output, (x, y), r, color, 2)
        bubble_count += 1
        print(f"Circle {i}: x={x}, y={y}, r={r}, fill_ratio={fill_ratio:.2f}")

# Save final image
cv2.imwrite(output_path, output)
print(f"✅ Total bubbles detected: {bubble_count}")
print(f"✅ Filled bubbles detected: {filled_count}")
print("✅ Result saved at:", output_path)

# Optional: show final result
cv2.imshow("Detected Bubbles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
