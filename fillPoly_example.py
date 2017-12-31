import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


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

# Read in the image
image = mpimg.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/exit-ramp.jpg')
# Display the image
plt.imshow(image)
plt.show()

# Specify the region of interest
left_bottom = [0, 540]
right_bottom = [960, 540]
left_top = [380, 270]
right_top = [580,270]
points = np.array([left_bottom, right_bottom, right_top, left_top])

new_image = region_of_interest(image, [points])
# Display the image
plt.imshow(new_image)
plt.show()