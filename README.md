Introduction:
	This project demonstrates one way to detect lane lines on a highway of any color from a video or image input file

Lane finding pipeline:

	An image is a huge two-dimensional matrix with 3 pixel values on each pixel denoting Red, Green and Blue. An image with resolution 720 X 580 indicates 580 rows and 780 columns of pixels of 3 channels. The top left corner of the image is chosen as the origin of the image with X increasing to the right and Y increasing downward.

Conversion to grayscale:

	Averaging the channel values on every pixel converts the image to grayscale. White corresponds to (255, 255, 255) and Black corresponds to (0, 0, 0). Every other color is a shade of gray. Converting to grayscale gives a single value on each channel.



Blurring the image:

	A sharp image will have sharply defined corners and noisy gradient. If we were to measure the slope of the pixel values there would be many peaks and valleys because of sharp gradients. Instead of dealing with noisy gradients if an image is blurred the edges/gradients become much smoother. Image processing becomes easier. Applying a gaussian blur is a standard way to do it. To apply a gaussian blur a kernel size is chosen. A two dimensional matrix is created with the kernel size and that filter is applied to the whole image such that introduces noise and softens the gradients

Canny Edge detection:

	A rise or fall of adjacent pixel values indicates an edge. A Canny edge detection looks for such a change in gradient and marks them as potential edges.



Hough Transformation:

	Hough transform is used to detect straight lines in an image. It transforms cartesian co-ordinates into parametric space where each line maps into a point and each point maps into a line and vice versa. By determining the number of intersections that happen in the parametric space straight lines in the cartesian space are detected.


Using the above mentioned ideas a lane building pipeline was created

Sequence of the pipeline:

Conversion to grayscale – provides single channel per pixel
Apply gaussian blur – this is to smoothen the gradients
Canny edge detection – Detection potential edges on the image
Select a region of interest – This clips a region from the image which is of interest. Since we are only interested in lane detection we need to focus only near the bottom half of the image. Imagine  a trapezium with the base as the bottom of the image and the top corners on either side of the image center.
Apply Hough transform on the region of interest – This will identify all the straight lines in the region of interest and that will be the detected lanes. The hough transform returns all the points that were detected and which were colinear.
Draw lines  - Once we have the co ordinates pf the lines that are colinear the points are connected to each other using straight lines and they are the final detected lane lanes from the image. They are colored red and overlaid on the original image

Enhancements to the draw lines function -

	If one side of the road has a solid lane and the other has dotted lines the basic draw lines function will return a solid line on the solid lane and an overlaying dotted line on the dotted lane. This does not make for a continuous lane and is not very useful. An enhancement could be made by having 2 solid lines on either side overlaying the lanes so they are continuous.

The way it was achieved in this project is the following

From the lines of the region of interest a line is extrapolated such that it covers the entirety of all the lines and lane.

Sort the lines detected by hough transform into the left half and right half of the image. This is done by measuring f the slope is positive or negative. Because the hough lines are from the region of interest only the lines corresponding to the lanes are passed in.  Using the equation of the straight line the bias term is calculated and stored for that particular line. An average slope of all the lines in the left half and right half is calculated. An average of the bias term on the left half and right half is calculated. This gives us a general slope and bias term per side of the image.

Next the minimum value of the x co ordinate is found by using the equation for a straight line. The bottom left of the region of interest’s y co-ordinate is known, average slope and average bias is known. Using these values x_min is calculated. The y co-ordinate will be the image’s row count as it is the bottom of the image. Similarly for the x co-ordinate for the top of image is calculated. The y co-ordinate for the top will be the top left y co-ordinate of the region of interest.
Finally a line is drawn from the top co-ordinates to the bottom co-ordinates which overlays the lane.
Same procedure is followed on the right half of the image


Further enhancements to the draw lines function:

	In order to make it even more robust the region of interest can further be divided into left region and right region. A parallellogram can be defined bounding the region where the lane lines are most likely to occur. That way the lines drawn will be less wiggly and noise free.

The cases where the current algorithm would fail is
when the car is not between the lanes to start with
when there is a lot of traffic and view of the lane is obstructed
there are other items near the lanes that are being picked up into the region of interest
