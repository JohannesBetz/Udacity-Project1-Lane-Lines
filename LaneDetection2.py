import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions2

import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip


images = os.listdir("test_images/")
for img_file in images:
    # print(img_file)
    # Skip all files starting with line.
    if img_file[0:4] == 'line':
        continue
    image = mpimg.imread('test_images/' + img_file)
    weighted = functions2.process_image(image)
    plt.imshow(weighted)
    # break
    mpimg.imsave('output_images/lines-' + img_file, weighted)

white_output = 'output_videos/white_output.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
a = clip1.fl_image
white_clip = clip1.fl_image(functions2.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'output_videos/yellow_output.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
a = clip1.fl_image
white_clip = clip1.fl_image(functions2.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(yellow_output, audio=False)

