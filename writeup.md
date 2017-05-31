# **Finding Lane Lines on the Road** 
---

In this project, you will use the tools you learned about in the lesson to identify lane lines on the road. You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below.
Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4". Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

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


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

### 2. Testing the pipline for Pictures and Videos


### 3. Potential Shortcomings

One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 4. Possible Improvements

A possible improvement would be to ...

Another potential improvement could be to ...


