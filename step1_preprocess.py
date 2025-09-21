import cv2
import os

# Use forward slashes or raw string
input_path = r"C:\Users\v-jananik\Downloads\OMR_Project\input\sample1.jpeg"
output_path = r"C:\Users\v-jananik\Downloads\OMR_Project\output\thresholded.jpg"

# Load image
image = cv2.imread(input_path)
if image is None:
    print("❌ Image not found! Put your OMR image in the input folder.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35, 15
)

# Save result
cv2.imwrite(output_path, thresh)
print("✅ Improved thresholded image saved at:", output_path)
