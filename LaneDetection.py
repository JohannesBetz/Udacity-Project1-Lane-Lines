
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions
import numpy as np
import os
import cv2

import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):

    imshape = image.shape
    gray = functions.grayscale(image)
    blur_gray = functions.gaussian_blur(gray, 5)
    canny_blur = functions.canny(blur_gray, 100, 200)
    vertices = np.array([[(50, imshape[0]), (450, 320), (500, 320), (900, imshape[0])]], dtype=np.int32)
    region_masked = functions.region_of_interest(canny_blur, vertices)
    hough_picture = functions.hough_lines(region_masked, 2, np.pi / 180, 20, 50, 30)

    result = functions.weighted_img(hough_picture, image)
    return result


################        IMAGES        #######################
images = os.listdir("test_images/")
for img_file in images:
    # print(img_file)
    # Skip all files starting with line.
    if img_file[0:4] == 'line':
        continue

    image = mpimg.imread('test_images/' + img_file)

    weighted = process_image(image)

    plt.imshow(weighted)
    # break
    mpimg.imsave('output_images/lines-' + img_file, weighted)


################        Videos       #######################
white_output = 'output_videos/white_output.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
a = clip1.fl_image
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'output_videos/yellow_output.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
a = clip1.fl_image
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'output_videos/extra.mp4'
clip2 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)



