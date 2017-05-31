#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions
import numpy as np
import cv2
import os

os.listdir("test_images/")


#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')
imshape = image.shape

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', imshape)

# 1. Convert the Picture to Greyscale
gray = functions.grayscale(image)
# plt.imshow(gray, cmap='gray')
# plt.show()

# 2. Apply Gaussian Blur
blur_gray= functions.gaussian_blur(gray, 5)
# plt.imshow(blur_gray, cmap='gray')
# plt.show()

# 3. Canny Edge Detection
canny_blur=functions.canny(blur_gray, 100, 200)
# plt.imshow(canny_blur, cmap='Greys_r')
# plt.show()

# 4. Region of Interest
vertices = np.array([[(50,imshape[0]),(450, 310), (500,310), (900,imshape[0])]], dtype=np.int32)
region_masked=functions.region_of_interest(canny_blur, vertices)
# plt.imshow(region_masked,cmap='Greys_r')
# plt.show()

# 5. Hough Lines Detection
hough_picture=functions.hough_lines(region_masked, 1, np.pi/180, 10, 50, 30)

# 7. Combine Pictures
combo = cv2.addWeighted(image, 0.8, hough_picture, 1, 0)
plt.imshow(combo)
plt.show()

