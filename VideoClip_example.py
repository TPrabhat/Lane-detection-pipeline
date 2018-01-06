# Import everything needed to edit/save/watch video clips

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# define region of interest globally
# the origin is the top left of the image
left_bottom = (50, 520)
right_bottom = (910, 520)
left_top = (430, 340)
right_top = (530, 340)


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

    """"
    The following routine has been modified to average out the draw lines function

    First the slope of each and every line from the frame is computed.
    The slope is separated into the left half and the right half od the image/video
    the bias, which is a constant, is computed as an average of all the parallel lines in one half of the image
    The x co-ordinates of the far min of the line and the far maximum of the line is calculated for each half odf the image
    A line is drawn from the far x,y to the far right x,y co ordinate
    """
    # define arrays to calculate
    slope_left = []
    slope_right = []
    c_left = []
    c_right = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y2-y1)/(x2-x1)
            if -1 < m < -0.3:
                slope_left.append(m)
                c_left.append(y1 - m * x1)
            elif 1 > m > 0.3:
                slope_right.append(m)
                c_right.append(y1 - m * x1)

    if len(c_left) > 0 and len(slope_left) > 0:
        x_min = (left_bottom[1] - np.nanmean(c_left))/np.nanmean(slope_left)
        x_left = (left_top[1] - np.nanmean(c_left))/np.nanmean(slope_left)
        cv2.line(img, (int(x_left), left_top[1]), (int(x_min), left_bottom[1]), color, thickness)

    if len(c_right) > 0 and len(slope_right) > 0:
        x_max = (right_bottom[1] - np.nanmean(c_right))/np.nanmean(slope_right)
        x_right = (right_top[1] - np.nanmean(c_right))/np.nanmean(slope_right)
        cv2.line(img, (int(x_right), right_top[1]), (int(x_max), right_bottom[1]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    img = image
    # Create a copy
    img_copy = np.copy(img)

    # Convert to grayscale
    gray = grayscale(img_copy)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    masked_edges = canny(blur_gray, low_threshold, high_threshold)

    points = np.array([left_bottom, right_bottom, right_top, left_top])

    clipped_image = region_of_interest(masked_edges, [points])

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 5
    max_line_gap = 3

    # Run Hough on edge detected image
    lines = hough_lines(clipped_image, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    result = weighted_img(lines, img_copy, 0.5, 1, 0)

    return result

# Read in the image
#test_image = mpimg.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test.jpg')
#combo = process_image(test_image)


#plt.imshow(combo)
#plt.show()

white_output = '/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test_videos_output/output.mp4'

clip1 = VideoFileClip("/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-videos/solidYellowLeft.mp4")
#clip1 = VideoFileClip("/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-videos/solidWhiteRight.mp4")
#clip1 = VideoFileClip("/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-videos/challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, codec = 'mpeg4', audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))