
import matplotlib.image as img
import matplotlib.pyplot as plt

image1 = img.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test.jpg')
image2 = img.imread('/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/exit-ramp.jpg')

#plt.subplot(221)
plt.imshow(image1)
plt.show()

#plt.subplot(212)
plt.imshow(image2)

plt.show()

import os
test_cases =  os.listdir("/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-images/")
print (test_cases)

for image in test_cases:
    test_image = img.imread("/home/prabhat/Downloads/pycharm-community-2017.3.1/bin/test-images/"+image)
    plt.imshow(test_image)
    plt.show()