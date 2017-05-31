# **Finding Lane Lines on the Road** 
---

In this project, you will use the tools you learned about in the lesson to identify lane lines on the road. You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below.
Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4". Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### 1. Pipline for detecting the Driving Lane

The tools we have provided from the udacity groupe are:

* color section
* grayscaling
* Gaussian Smoothing
* Canny Edge Detection
* Hough Transform Line Detection

The  goal was to piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display. Once we figured out a pipline for processing an image, the goal was to apply this process to different videos

My pipeline consists of the definition of an function file, where every function that is needed is included. The function files is called FUNCTIONS.py.

My Pipline for Processing a single picture is integrated in the Function "def process_image"

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

First, the shape of the image is figured out to get the heigth and weidth of the picutre. With the grayscale function, the picture is tourned into gray colors, after that the picture will be blurred with the gaussian blur function. This is the preprocessing befor the Canny function can be applied to the picture. The canny function detects the edges in the picture and helps to figure out the step between a line on the road and the road. Befor we find the lines we have to figure out a frame in the picture, where we expect the lanes. We are doing that with the region_masked function where we define a polygon for the lane detection. The last step is the hough line detection, where we converting our image in the hough space. In this function we try to find a line through all the pixels on the road line segment. When we have found the lines, we can draw them into the picture.

The hough alogirhtm gives back points for a lot of lines that are drawn in the picture. So right now, we would have a lot of different lines, but we just want one line. In order to to that I modified the draw_lines() function by adding a calculation for the slope for left and right lines. After that i wanted to figure out just one line, that is the average line out of all these lines. With this algorithm i just get 4 points, 2 for the left lane and two for the right one.


    
    def draw_lines1(img, lines, color=[255, 0, 0], thickness=8):
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
    top = 320
    bottom = 550
    left_x1s = []
    left_y1s = []
    left_x2s = []
    left_y2s = []
    right_x1s = []
    right_y1s = []
    right_x2s = []
    right_y2s = []
    for line in lines:
        # print(line)
        # Feel this is the brute force method, but I'm time constrained. I will research ideal numpy methods later.
        for x1, y1, x2, y2 in line:

            x_values = float(x2 - x1)
            y_values = float(y2 - y1)
            slope = float(y_values / x_values)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue
                left_x1s.append(x1)
                left_y1s.append(y1)
                left_x2s.append(x2)
                left_y2s.append(y2)
            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue
                right_x1s.append(x1)
                right_y1s.append(y1)
                right_x2s.append(x2)
                right_y2s.append(y2)

    try:
        avg_right_x1 = int(np.mean(right_x1s))
        avg_right_y1 = int(np.mean(right_y1s))
        avg_right_x2 = int(np.mean(right_x2s))
        avg_right_y2 = int(np.mean(right_y2s))
        right_slope = get_slope(avg_right_x1, avg_right_y1, avg_right_x2, avg_right_y2)

        right_y1 = top
        right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
        right_y2 = bottom
        right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    try:
        avg_left_x1 = int(np.mean(left_x1s))
        avg_left_y1 = int(np.mean(left_y1s))
        avg_left_x2 = int(np.mean(left_x2s))
        avg_left_y2 = int(np.mean(left_y2s))
        left_slope = get_slope(avg_left_x1, avg_left_y1, avg_left_x2, avg_left_y2)

        left_y1 = top
        left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
        left_y2 = bottom
        left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass
        
    def get_slope(x1, y1, x2, y2):
     return float(float(y2 - y1) / float(x2 - x1))

After drawing the lines, the original image and the two lines from the drawing function are combined again, you can see two red lines in the picture afterwards

### 2. Testing the pipline for Pictures and Videos

To test the pipline i applied it to different pictures and videos. Down below you find the algorithm for including loading and postprocessing the pictures and videos
    
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
    

[image1]: ./output_images/lines-solidWhiteCurve.jpg "solidwhite"


### 3. Potential Shortcomings

My code failed in the detection of the challenge video. I think this might be because of the road conditions (anything that changes/obscures the lanes will result in incorrect (or no) lane lines being detected. For example, snow, heavy rain, bright sunlight, or even a wide car/truck that (partially) obscures the lines. 

Second, there might be problems when the car position is changing: moving the car from left to right inside the same lane, changing lanes. I think the algorithm right now i just for straight lines, so the curve detection in the video failed.

### 4. Possible Improvements

Possible improvements
1. Improve the "smoothing algorithm" by something more sophisticated than simply taking the weighted average of the current and previous frame's lane line
2. Dynamically change the region of interest in response to changes in the position of the car (e.g. pitch changes when going up/down a steep hill)
3. Dynamically change the various parameters: while current parameter values work ok for the set of images and videos provided with the project, it is most likely a classic case of overfitting. Canny and Hough parameters, the color mask values, even the blur kernel size, can/should probably be set dynamically depending on road conditions, weather etc
4. Color Detection: Right now i just use the Canny Detection for the Edge, i should combine it with the color.




