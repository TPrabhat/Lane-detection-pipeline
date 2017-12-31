#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/exit-ramp.jpg')
plt.subplot(221)
plt.imshow(image)


gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#plt.subplot(222)
#plt.imshow(gray)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
#plt.subplot(223)
#plt.imshow(blur_gray)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.subplot(222)
plt.imshow(edges, cmap='Greys_r')

low_threshold = 50
high_threshold = 250
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.subplot(223)
plt.imshow(edges, cmap='Greys_r')

low_threshold = 10
high_threshold = 350
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.subplot(224)
plt.imshow(edges, cmap='Greys_r')

plt.show()