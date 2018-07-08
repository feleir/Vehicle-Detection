# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

[//]: # (Image References)
[car_hog_features_channel_0]: ./output_images/car_hog_features_channel_0.png
[car_hog_features_channel_1]: ./output_images/car_hog_features_channel_1.png
[car_hog_features_channel_2]: ./output_images/car_hog_features_channel_2.png
[non_car_hog_features_channel_0]: ./output_images/non_car_hog_features_channel_0.png
[non_car_hog_features_channel_1]: ./output_images/non_car_hog_features_channel_1.png
[non_car_hog_features_channel_2]: ./output_images/non_car_hog_features_channel_2.png
[heat_map]: ./output_images/heat_map.png
[car_histogram_features]: ./output_images/car_histogram_features.png
[non_car_histogram_features]: ./output_images/non_car_histogram_features.png
[hog_subsampling]: ./output_images/hog_subsampling.png
[hot_windows]: ./output_images/hot_windows.png
[sliding_windows]: ./output_images/sliding_windows.png
[test_image]: ./output_images/test_image.png
[test_images_processing]: ./output_images/test_images_processing.png
[output_gif]: ./video_output.gif

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third cell of the IPython notebook `vehicle_detection.ipynb`, specifically the function `single_image_features` which will extract, depending on the passed parameters:

- Spatial features
- Histogram features
- Hog features

I started by reading in all the `vehicle` and `non-vehicle` images. (cell #2).
```
Found 8968 non-vehicle images.
Found 8792 vehicle images.
```

Then started experimenting with HOG, special and histogram parameters, after few rounds of testing I ended up with the following ones (cell #4)

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 128    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, 700] # Min and max in y to search in slide_window()
```

Here is an example of one of each of the `vehicle` and `non-vehicle` raw and normalized features, using and `StandardScaler`.

![Car histogram raw and normalized features][car_histogram_features]
![Non car histogram raw and normalized features][non_car_histogram_features]

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

#### Car
![Car hog features channel 0][car_hog_features_channel_0]

![Car hog features channel 1][car_hog_features_channel_1]

![Car hog features channel 2][car_hog_features_channel_2]

### Non car
![Non car hog features channel 0][non_car_hog_features_channel_0]

![Non car hog features channel 1][non_car_hog_features_channel_1]

![Non car hog features channel 2][non_car_hog_features_channel_2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I choosed the final HOG parameters by testing with the example images to help determine which combination describes better the original image, also checking the accuracy of the `SVM` classifier for validation.

I decided to use `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` which provided good results and help reduce positive detections. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear `SVM` classifier (cell #8), using the parameters described before.
- `YCrCB` color space
- All Hog channels, `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
- Spatial features, `spatial_size=(16,16)`
- Histogram features, `hist_bins=128`

The training set was shuffled and splitted in 80% training and 20% testing, and traing with an accuracy of `0.9904`

```
Feature vector length: 6444
17.14 Seconds to train SVC...
Test Accuracy of SVC =  0.9904
```

Cause the accuracy was excellent I decided to not use `GridSearchCV` to improve it even further but the code to do this can be found in cells #9 and #10.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started first with a the `slide_window` function described in the class, choosing different window sizes.

```
window_sizes = [(320, 320), (256, 256), (128, 128), (96, 96), (64, 64)]
test_windows = []
for window_size in window_sizes:
    step_windows = slide_window(sample_test_image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=window_size, xy_overlap=(0.5, 0.5))
    test_windows = test_windows + step_window
```

![Sliding windows][sliding_windows]

But then using `Hog subsampling`, which can be checked in then `find_cars` function, which gave much better results and performance.
For the video pipeline I selected different scales to apply the sliding windows technique (cell #22 `process_video_image` function).

```
scales = [1, 1.5, 2, 2.5, 3]
    # Apply sliding windows technique for HOG features to compute
    for scale in scales:
        hot_windows = hot_windows + find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
```

![Hog subsampling][hog_subsampling]

### Duplicate detection and bound boxes

#### 1. Describe how you implemented a filter to detect false positives

I created a heatmap and then thresholded that map to identify vehicle positions, using a combination of the functions `add_heat` and `apply_threshold`.
Then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, assuming each blob is a vehicle and creating a bounding box using the function `draw_label_bboxes`.

![Heat Map][heat_map]

### Pipelin processing

#### 1. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Test images][test_images_processing]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To be able to do this I created a cache of the hot windows found in the previous frames, up to `10` frames before the current one, after combining all of then I processed the bound boxes with a threshold of `7` which helps reduce false positives, applying the heat-map process described before.

```
previous_bboxes = collections.deque(maxlen=10)
....
    # Add to previous bboxes
    previous_bboxes.append(bbox_list)
    # Get all hot boxes
    avg_bbox_list=[]
    for box in previous_bboxes:
        for b in box:
            avg_bbox_list.append(b)
    # Smoothing - Apply avg heatmap to further reduce false positives and smooth the bounding boxes
    bbox_list, heatmap = process_bboxes(img,avg_bbox_list,threshold=avg_threshold,show_heatmap=True)   
```

After that processing I added the found heatmap on the top-right corner of each frame applying the function `add_heatmap_to_image`, the results can be checked in the video or in the following gif.

![Result](https://github.com/feleir/Vehicle-Detection/raw/master/video_output.gif)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the major improvements will be to identify multiple cars when they are next to each other, the heatmap technique will create the same bounding box for all of them so prediction is not satisfying. I think one of the ideas to avoid this is to combine car detection with lane detection techniques so the bounding boxes can be identified properly.

### ToDos

I didn't have time to combine lane and vehicle detection in the same pipeline, will try to do this exercise in the future.

