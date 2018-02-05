# Vehicle Detection and Tracking

In this project I will built a pipeline which detects and tracks vehicles in a video. 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The way this is organized is I do each of the above steps separately with an example or to, and then combine the whole thing into a pipeline at the end. 

---

## Loading Data 

Some examples were provided to test the functions on in the project folder. I loaded up the 6 images from the test images folder. Also, I loaded in the data for training the classifier here.

