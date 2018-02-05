# Vehicle Detection and Tracking


[//]: # (Image References)

[image1]: ./output_images/car0.png "Example Car"
[image2]: ./output_images/car7.png "Example Car"
[image3]: ./output_images/notcar0.png "Example not car"
[image4]: ./output_images/notcar7.png "Example not car"
[image5]: ./output_images/HOG0.png "Example HOG feaures"
[image6]: ./output_images/HOG7.png "Example HOG features"
[image7]: ./output_images/color_hist11.png "Histograms of Color Features"
[image8]: ./output_images/spatial_hist11.png "Histograms of Spatial Feautres"
[image9]: ./output_images/feature_histograms.png "Histograms of Combined Features"
[image10]: ./output_images/all_windows.png "Simple window search"
[image11]: ./output_images/simple_detect0.png "Simple Classification"
[image12]: ./output_images/simple_detect5.png "Simple Classification"
[image13]: ./output_images/heatmap0.png "Heatmap example"
[image14]: ./output_images/heatmap5.png "Heatmap example"
[image15]: ./output_images/final_out0.png "Final result example"
[image16]: ./output_images/final_out5.png "Final result"

[image17]: ./test_images/test0.png "Test image"
[image18]: ./test_images/test4.png "Test image"

[image19]: ./output_images/complex_detect0.png "Test detect"
[image20]: ./output_images/complex_detect5.png "Test detect"

[image21]: ./output_images/final_out4.png "Final result"


In this project I will built an image pipeline which detects and tracks vehicles in a video. 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The way this is organized is I do each of the above steps separately with an example or to, and then combine the whole thing into a pipeline at the end. For the piece by piece code walkthrough, launch the Ipy notebook:
[Ipython notebook: Vehicle Detection](https://github.com/andrewsommerlot/Carnd-Vehicle-Detection/blob/master/vehicle_detection_tracking.ipynb) 

---

## Training and Example Data 

Some examples were provided to test the functions on in the project folder. I loaded up the 6 images from the test images folder. Also, I loaded in the data for training the SVM classifier. 

![Test image 1][image17]

**Example of test image I used while bulding the pipeline**




![test image 2][image18]

**Example of test image I used while bulding the pipeline.**


The data used for training the SVM classifier is available [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) for pictures containing vehicles, and [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) for pictures not contianing vehicles. The data are a combination of pictures available in the GTI vehicle image database, the KITTI vision benchmark suite, and examples from the project video. Here are a few examples of car and not car pictures from the training data. After loading in the data, I found there were 8968 examples of not cars, and 8792 examples of cars. Although this is in no way a complete data analysis, I decided to move on with building the pipeline. 


![Car Example][image1]

**Example of a car from the training data**



![Car Example][image2]

**Example of a car from the training data**



![Not car][image3]

**Example of not a car from the training data**



![Not Car][image4]

**Example of not a car from the training data**


## Histogram of Oriented Gradients (HOG) feature extraction

I extrated the HOG feature from images. A lot of this code is from the lessons, but required some tweaking for exploration. HOG features extraction takes a look at pieces of each image and counts up a gradient orientation, or direction in that piece. Thats why they HOG feature pictures look like groups of little arrows. These can be very helpful object detection. Here are a couple of examples. 

![HOG Example][image5]

**HOG Feature from an example training data**



![HOG Example][image6]

**HOG features from an examle training data**


  

## Additional Feauture Extractions

Additionally, I extracted spatial and color histogram features from training data. Much of this code could be found in the lessons as well, again I applied some tweaks. The color histogram feature is created for an image by simple taking histograms of each channel slice separately and then concatonating. The spaitial binning extraction combines the image data while resizing it to a specifed shape. Below are histograms of each type of these extra features from some randomly selected training data. 


![Spatial Binning][image7]

**Histograms of spaital binning feature extractions**



![Color histograms][image8]

**Histograms of color binning feature extractions**




I had to standardize these features later as they are all combined. Here is a few histograms of the combined features, showing the new range.

![Combined and Standardized histograms][image9]

**Histograms of combined and standardized features**




## Implement a sliding-window technique

Now, the trained classifier has to be implemented through a sliding window technique. The sliding window technique will select various "boxes" or subsets of an imput image which will be transformed into inputs for the classifier. This way, the classifier can be applied over and over to all areas of an input image. In a live application, for each window shown below, the pipline needs to extract the features specified in the above sections, and then run them through a trained classifier to get a car or no car class. I will actually search alot more windows than this to get overlapping classifications, but the idea is the same. 

The window search dictates where the feature extraction function will pull feaure vectors out of to standardize, concatonate, and feed into the classifier I'll train in the next step. 

Below is an output of a simple window-window search process, where a classifier can be run on each window to look for images containing cars. 

![Simple window search][image10]

**Grid of simple window search, for each window, features must be extracted and a fed through a trained classifier**




## Training a classifier

I had to train a classifier on privided training data. Although these data are considerably clean, they come from different sources and may cause a few problems. For now, I'm just using them all together with the precautions of shuffling and splitting into train (80%) and test (20%) data. The GTI data are time series and could cause overfitting from nearly identical examples in both train and test, but I'm forging on for now. 

There are a number of parameters to adjust in the pipeline now. To train the classifier, after many round of trial and error training the SVM I used the following parameters for feature extraction: 

|Color Space| LUV |
|HOG orientation directions| 16 |
|Pixels per cell| 8 |
|Cells per block| 4 | 
|HOG channel| 0 |
|Spatial Size| 16x16 | 
|Historgram bins| 32 |

These parameters can greatly train what the extracted vector looks like that the SVM will be trained on. All of these feautures were standardize and shuffled before being split in to train and test sets. For the classifer, I ended up using the default values for linearSVM in scikit learn. After a few minutes, I got an accuracy of about 0.98. I had some fears of overfitting at this point, but I moved on. 

I tested the simple sliding window technique with the feature extraction and classifer pipline. A couple of examples are below. The process is alright, but needs alot more search windows and post processing to be more reliable, as there are a number of false positives that will cause problems in the pipeline result. 

![Simple Vehicle Detection][image11]

**Example of simple vehicle detection process with no cars**




![Simple Vehicle Detection][image12]

**Example of simple vehicle detection process with no cars**




## Further Sliding Window Techniques to Improve Performance

The classifier is ok, but still needs some work before implementing on a movie. Here I'll continue to use sliding window techniques to improve the result. Using the find_cars function from class material, I was able to serach over the image more times with more window variations. The resulting sliding window process produces many more chances for the classifier to find cars, which can give the whole process a better chance to find any cars in the images. However, it does have the drawback of creating more opportunities for false postives. Below are two examples which illitrate this. 


![Complex Vehicle Detection][image19]

**Example of complex sliding window vehicle detection process with no cars**




![Complex Vehicle Detection][image20]

**Example of complex sliding window vehicle detection process with no cars**




## Dealing with False Positives and Multiple Detections

There are still false positives to deal with. I implemented a heat map thresholding process to get rid of a lot of these false positives. The following fucntions quantify how often an area is detected through overlapping classification boundaries. Then, a threshold is applied to remove those areas with less overlapping postive classifications. This removes some of the extraneous or "wild" classifications and makes sure if boundaries are printed out to the output image they have to have been classified as positive multiple times. The heatmap process also has the effect of combining multiple detections into a single detection. This is important as printing out all the detections, such as seen in the figures above, is messy. With heatmap thresholding, a better final estimate of car locations is made with higher confidence.


![Complex Vehicle Detection][image13]

**Heatmap vehicle detection process with no cars. Due to the threshold, the false positive from the previous step has dissapeared**




![Complex Vehicle Detection][image15]

**Example of final detection based off above heatmap. This detection does not have the false positive it had before**




![Complex Vehicle Detection][image14]

**Heatmap vehicle detection process with cars. The threshold process also aggregates multiple detections, making a cleaner final detection**




![Complex Vehicle Detection][image16]

**Example of complex sliding window vehicle detection process with no cars**




## Building the pipeline and testing

Next I put these functions together into a pipeline is the pipeline built with the functions above and made to output a tracking image based on one input image. There are a good number of parameters hard coded in here by now, so it was a good idea to double check the pipeline output was as expected. The output is not perfect, but I think its worth a try.  


![Complex Vehicle Detection][image15]

**Vehicle detection pipeline output on example**




![Complex Vehicle Detection][image21]

**Vehicle detection pipeline output on example**




## Run the pipeline on a video 

Next I'll use the pipeline to create a video. Much like the previous project, I'm using moviepy.editor to loop the frames and create a new video. The process loops the detection pipeline over the frames in the video and outputs a new video with the overlayed bounding boxes.

[![Advanced Lane Lines](http://img.youtube.com/vi/https://youtu.be/WNPw9d1A9Jo/0.jpg)](https://youtu.be/WNPw9d1A9Jo "Vehicle Detection and Tracking")


## Discussion and Conclusion

The pipeline performed adaquitely, but fell short of nice, smooth detections completely absent of false positives. More post processing would help smooth out the bounding boxes. Multiple aggregation and false positve detection methods could be employed as an ensemble of information to inform the bounding boxes. Previous box sizes could be stored, and next box sizes compared, as we know the size of the car in the video should not change very fast if they are going usual speeds. Additionally, the hard selection of one color space will likely not be robust in various lighting and weather scenarios. My process is also pretty slow, which would be a big problem in live implementation. Overall, this pipeline demonstrates a process which uses hand extracted features to detect vehicles in a video with reasonable accuracy but could definately be improved. 


