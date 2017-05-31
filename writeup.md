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

The hough alogirhtm gives back points for a lot of lines that are drawn in the picture. So right now, we would have a lot of different lines, but we just want one line. In order to to that I modified the draw_lines() function by adding a calculation for the slope for left and right lines (based on the work of https://github.com/eosrei/CarND-P01-Lane-Lines). With this calculation i sorted the points for the left lane and the right lane. After that, i made a linear regression to get the line which goes through all the points. This line has the parameter m and b of the line and with them i can calculate the max and min points for every line.

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
    
    imshape = img.shape

    vertices = np.array([[(50, imshape[0]), (450, 320), (500, 320), (900, imshape[0])]], dtype=np.int32)
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    left_x1_max = 200
    left_y1_max = imshape[0]
    left_x2_max = 450
    left_y2_max = 320

    right_x1_max = 500
    right_y1_max = 320
    right_x2_max = 870
    right_y2_max = imshape[0]


    for line in lines:
        for x1, y1, x2, y2 in line:

            x_values = float(x2 - x1)
            y_values = float(y2 - y1)
            slope = float(y_values / x_values)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue

                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)

            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)

    right_x.append(right_x1_max)
    right_x.append(right_x2_max)
    right_y.append(right_y1_max)
    right_y.append(right_y2_max)

    left_x.append(left_x1_max)
    left_x.append(left_x2_max)
    left_y.append(left_y1_max)
    left_y.append(left_y2_max)

    x_right = np.asarray(right_x)
    y_right = np.asarray(right_y)
    A_right = array([x_right, ones(len(x_right))])
    w_right = linalg.lstsq(A_right.T, y_right)[0]  # obtaining the parameters
    line_right = w_right[0] * x_right + w_right[1]  # regression line

    x_left = np.asarray(left_x)
    y_left = np.asarray(left_y)
    A_left = array([x_left, ones(len(x_left))])
    w_left = linalg.lstsq(A_left.T, y_left)[0]  # obtaining the parameters
    line_left = w_left[0] * x_left + w_left[1]  # regression line

    x1_right = min(x_right)
    x2_right = int((imshape[0] - w_right[1])/w_right[0])
    y1_right = int(min(line_right))
    y2_right = imshape[0]

    x1_left = int((imshape[0] - w_left[1])/w_left[0])
    x2_left =max(x_left)
    y1_left = imshape[0]
    y2_left = int(min(line_left))


    cv2.line(img, (x1_right,y1_right),(x2_right,y2_right), color, thickness)
    cv2.line(img, (x2_left, y2_left), (x1_left, y1_left), color, thickness)
    

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

I think my code works probably mostly for straigth lines

My code failed in the detection of the challenge video. I think this might be because of the road conditions (anything that changes/obscures the lanes will result in incorrect (or no) lane lines being detected. For example, snow, heavy rain, bright sunlight, or even a wide car/truck that (partially) obscures the lines. 

Second, there might be problems when the car position is changing: moving the car from left to right inside the same lane, changing lanes. I think the algorithm right now i just for straight lines, so the curve detection in the video failed.

### 4. Possible Improvements

Possible improvements
1. Improve the linear regression to an polyomal regression for the curves
2. Right now i am using a defined window to calculate the parameters, i think there should be an anctive change of this parameters because right now thei are just constants - Dynamically change the region of interest in response to changes in the position of the car 
3. Dynamically change the various parameters: while current parameter values work ok for the set of images and videos provided with the project, it is most likely a classic case of overfitting. Canny and Hough parameters, the color mask values, even the blur kernel size, can/should probably be set dynamically depending on road conditions, weather etc
4. Color Detection: Right now i just use the Canny Detection for the Edge, i should combine it with the color.




