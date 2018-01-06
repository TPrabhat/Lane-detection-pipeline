# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    slope = []
    slope_left = []
    slope_right = []
    minx, maxx = 0, 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            slope.append((y2-y1)/(x2-x1))
            minx, maxx = np.amin(lines), np.amax(lines)
    for idx in range(len(slope)):
            if slope[idx] > 0:
                slope_right.append(slope[idx])
            else:
                slope_left.append(slope[idx])

    x_left = int(minx - (left_bottom[1] - left_top[1])/(np.average(slope_left)))
    x_right = int(maxx - (left_bottom[1] - left_top[1])/(np.average(slope_right)))
    cv2.line(img, (x_left, left_top[1]), (minx, left_bottom[1]), color, thickness)
    cv2.line(img, (x_right, right_top[1]), (maxx, left_bottom[1]), color, thickness)

    #print(minx, maxx, x_left, x_right, np.average(slope_right), np.average(slope_left))
    #plt.imshow(img)
    #plt.show()


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    #print (lines.shape)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α, β, λ):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Read in the image
#image = mpimg.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-images/solidWhiteRight.jpg')
image = mpimg.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-images/exit-ramp.jpg')
# Display the image
#plt.subplot(121)
#plt.imshow(image)
#plt.show()
# Create a copy
img_copy = np.copy(image)

# Convert to grayscale
gray = grayscale(img_copy)
#plt.subplot(332)
#plt.imshow(gray)
#plt.show()

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = gaussian_blur(gray, kernel_size)
#plt.subplot(333)
#plt.imshow(blur_gray)
#plt.show()

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = canny(blur_gray, low_threshold, high_threshold)
#plt.subplot(334)
#plt.imshow(masked_edges)
#plt.show()

# Specify the region of interest
left_bottom = (0, 540)
right_bottom = (960, 540)
left_top = (430, 320)
right_top = (530, 320)

points = np.array([left_bottom, right_bottom, right_top, left_top])

clipped_image = region_of_interest(masked_edges, [points])
# Display the image
#plt.subplot(335)
#plt.imshow(clipped_image)
#plt.show()

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 5
max_line_gap = 3

# Run Hough on edge detected image
lines = hough_lines(clipped_image, rho, theta, threshold, min_line_length, max_line_gap)
#plt.subplot(132)
#plt.imshow(lines)
#plt.show()

# Draw the lines on the edge image
combo = weighted_img(lines, img_copy, 0.8, 1, 0)
#plt.subplot(122)
plt.imshow(combo)
plt.show()