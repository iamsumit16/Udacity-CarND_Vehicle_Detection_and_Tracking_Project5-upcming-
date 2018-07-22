## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I randomly select two car images and two non-car images to compare there HOG figures. The two example images are shown as follows, the parameters of the HOG feature I am using is orient = 9 pix_per_cell = 8 cell_per_block = 2. We can find that the HOG features of the cars and non-cars are really very different from each other. We can still tell the outline of the car from the HOG image.


![alt text](https://github.com/iamsumit16/Vehicle_Detection_and_Tracking_Project5/blob/master/images/hog.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

Among the various combinations of parameters, I chose the following set based on best accuracy:'

color_space = 'YCrCb'  
spatial_size = (32, 32) 
hist_bins = 32  
orient = 9  
pix_per_cell = 8  
cell_per_block = 2  
hog_channel = 'ALL'  
spatial_feat = True  
hist_feat = True  
hog_feat = True`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The extracted features where fed to LinearSVC model of sklearn with default setting of square-hinged loss function and l2 normalization. The trained model had accuracy of 99.93% on test dataset.

The trained model along with the parameters used for training were written to a pickle file to be further used by vehicle detection pipeline.

Initially I tried the GridSearchCV to find the best combination of kernal and parameter 'C' but that takes a lot of time on my machine to run everytime. I may include in the future update of the project.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

he bacis function 'find_cars()' to detect the car was used for this. It is used to extract features using hog sub-sampling and make predictions. The hog sub-sampling helps to reduce calculation time for finding HOG features and thus provided higher throughput rate.

In the basic one, 64 was the orginal sampling rate, with 8 cells and 8 pix per cell. The step size is cells_per_step = 2, which means instead of overlap, we shift 2 cells each step.

The following is the result when I set the scale to be scale = 1.5, the detection example.


![alt text](https://github.com/iamsumit16/Vehicle_Detection_and_Tracking_Project5/blob/master/images/sliding15.png)


Then I used the heat map operation to take care of the multi-detection and reduce the false positive. The example images are shown below, which is basicly good.


![alt text](https://github.com/iamsumit16/Vehicle_Detection_and_Tracking_Project5/blob/master/images/heatmap.png)


I have tried to directly use the one search scale scale = 1.5 with heat map with threshold = 1 to build the pipeline for video. This pipeline can be found in the function detect_vehicles(). The output video is okayish. However, the there are still some false positives shown up and sometimes. And the bounding boxes are not stable and the cars in some frame may not be detected.

In order to solve these problems, I decided to use the three scales search windows.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. I record the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

The method find_cars_smooth() is used to detect the car. It is basicly the same as the function find_cars() defined before. However, it allows the multi-scale search. More importantly, the search is optimized by processing complete frames only once every 10 frames. The restricted search is performed by appending 50 pixel to the heatmap found in last three frames. It really helps a lot to make the detection more robust and stable.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_output_smooth.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

#### 3. Resulted heatmaps from the pipeline:

![alt text](https://github.com/iamsumit16/Vehicle_Detection_and_Tracking_Project5/blob/master/images/heatmapss.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that in some frame of the video, there will be missed detection. I think I need to enlarge the search area. However, large area will take more time to process the image. The reason that we want to use the multi-scale search is to avoid false positive. Because the linear SVC classifier may not be the best to choose in terms of time and accuracy. I think we may want to use some better classifiers, such as the CNN or SVM with non-linear kernals.

The other issues I face are: When there is a car behind another, the resulted boxes combine into one, its really hard to catch the small cars with this search, and processing speed of frames per second is really slow.

