"""
Road Lane Detection using OpenCV
"""

# Importing Libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as pimg
import math
import cv2

# Reading Image
img = pimg.imread('Images/Image.jpg')
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img)
plt.show()

# Converting to Grayscale
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("Grayscale Image")
plt.imshow(img, cmap='gray')
plt.show()

# Gaussian Image Smoothing
"""
It averages out anomalous gradients in the image.
"""
def Gaussian_Blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

img = Gaussian_Blur(img, 7)
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("After Gaussian Blur")
plt.imshow(img, cmap='gray')
plt.show()

# Applying Canny Edge Detection
"""
Canny Edge Detection is an operator that uses the horizontal and vertical gradients of the pixel values of an image to detect edges.
"""
def Canny_Edge_Detection(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

img = Canny_Edge_Detection(img, 50, 250)
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("After Canny Edge Detection")
plt.imshow(img, cmap='gray')
plt.show()

# Region of Interest
"""
The region of interest for the car’s camera is only the two lanes immediately in it’s field of view and not anything extraneous. 
"""
def Region_of_Interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

lowerLeftPoint = [0, 150]
lowerRightPoint = [350, 150]
upperLeftPoint = [0, 65]
upperRightPoint = [350, 65]
vertices = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)

img = Region_of_Interest(img, vertices)
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("After selecting Region of Interest")
plt.imshow(img, cmap='gray')
plt.show()

# Drawing Lines
"""
This function draws lines with a color and thickness.
"""
def Draw_Lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Hough Lines
def Hough_Lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    img: Should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    Draw_Lines(line_img, lines)
    return line_img

img = Hough_Lines(img, 1, np.pi/180, 30, 20, 20)
# Displaying Image
plt.figure(figsize=(6,6))
plt.title("After Hough_Lines")
plt.imshow(img, cmap='gray')
plt.show()
