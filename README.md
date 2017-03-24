#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[image1]: ./report-images/01-dataset-summary.png "Dataset Summary"
[image2]: ./report-images/02-dataset-preprocessing.png "Dataset Pre-processing"
[image3]: ./report-images/03-web-dataset.png "German traffic signs from the web (GTSfW)"
[image4]: ./report-images/04-web-dataset-preprocessing.png "GTSfW Pre-processing"
[image5]: ./report-images/05-conv1.png "conv1"
[image6]: ./report-images/06-conv1_relu.png "conv1_relu"
[image7]: ./report-images/07-conv1_pool.png "conv1_pool"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README
####1. Writeup / README includes all the rubric points.

Here is a link to my [project code](https://github.com/olpotkin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

####1. Basic summary of the data set.

The code for this step is contained in the code cell **#2** of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

####2. Exploratory visualization of the dataset.

The code for this step is contained in the code cell **#3** of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a table structured images with a 4 parameters:

* Sign's class name
* Number of occurrences in Training set
* Number of occurrences in Validation set
* Number of occurrences in Test set

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell **#4** of the IPython notebook.

As a first step, I decided to **convert the images to grayscale** because in case with traffic signs, the color is unlikely to give any performance boost. But for natural images, color gives extra information, and transforming images to grayscale may hurt performance.

As a second step, I **normalize the image** data to reduce the number of shades to increase the performance of the model.

Here is an example of a traffic sign image before and after each step of preprocessing.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the code cell **#1** of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. There is different information from many sources about proportion of examples in the sets, but average is around 60-70% for training set, 10-15% for validation set, 20-25% for test set.

My final training set had **69598** number of images. My validation set and test set had **4410** and **12630** number of images.

The code cell **#4** of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because it's a common technique to improve model's precision. To add more data to the the data set, I used the following techniques to original images:

* 10 degree rotation counterclock-wise
* Zoom (ratio = 1.25)

####3. Final model architecture.

The code for my final model is located in the code cell **#6** of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 		|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 						|
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16 	|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 						|
| Flatten	        	| outputs 400 										|
| Fully connected		| outputs 120  										|
| RELU					|													|
| Dropout				| keep probability = 0.75 							|
| Fully connected		| outputs 84  										|
| RELU					|													|
| Dropout				| keep probability = 0.75 							|
| Fully connected		| outputs 43 logits  								|
|						|													|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the code cells **#5**, **#8** of the ipython notebook. 

To train the model, I used the follow global parameters:

* Number of epochs = 10. Experimental way: increasing of this parameter doesn't give significant improvements.
* Batch size = 128
* Learning rate = 0.001
* Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
* Dropout = 0.75 (for training set only)

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 

The code for calculating the accuracy of the model is located in the code cell **#9** of the Ipython notebook.

My final model results were:

* training set accuracy of **0.99303**
* validation set accuracy of **0.94580**
* test set accuracy of **0.92605**

This solution based on modified LeNet-5 architecture. With the original LeNet-5 architecture, I've got a validation set accuracy of about 0.88.

Architecture adjustments:

* Step 1: perform preprocessing (grayscale and normalization). Results for training and validation sets on epoch #10 were 0.99253 and 0.89435 that's mean overfitting.
* Step 2: generate augmented training data. Results for training and validation sets on epoch #10 were 0.98437 and 0.91633 that's mean overfitting again.
* Step 3: Important design choices - **apply dropout** (A simple way to prevent neural networks from overfitting). Results for training and validation sets on epoch #10 were around 0.99237 and 0.93673 (keep_prob values were in range 0.5-0.8).

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3]

The image #1 might be difficult to classify because it's rotated

The image #2 might be difficult to classify because it has complex background

The image #3 might be difficult to classify because it has noizy background

The image #4 might be difficult to classify because it has unusual color of sign's border (looks like pink)

The image #5 might be difficult to classify because it has noizy background

The image #6 might be difficult to classify because it has noizy background

The image #7 might be difficult to classify because background contains 2 parts

The image #8 might be difficult to classify because it's rotated

Here is example of German traffic sign after preprocessing:

![alt text][image4]

####2. The model's predictions on new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the code cells **#14**, **#15** of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution  		| General caution   							| 
| No passing   			| No passing 									|
| Speed limit (20km/h)	| Speed limit (20km/h)							|
| Traffic signals		| General caution				 				|
| Stop  				| Stop     										|
| Priority road			| Priority road      							|
| Yield 				| Yield      									|
| Turn right ahead		| Turn right ahead      						|

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of **87.5%**. This compares favorably to the accuracy on the test set of **92.6%**.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell **#16** of the Ipython notebook.

For the **image #1**, the model is relatively sure that this is a **General caution sign** (probability of 0.99843), and the image does contain a **General caution sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99843         		| General caution   							| 
| .00135     			| Traffic signals 								|
| .00011				| Pedestrians									|
| .00008	      		| Road work					 					|
| .00001				| Road narrows on the right      				|


For the **image #2**, the model is relatively sure that this is a **No passing sign** (probability of 1.0), and the image does contain a **No passing sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| No passing   									| 
| 0.     				| No passing for vehicles over 3.5 metric tons 	|
| 0.					| Vehicles over 3.5 metric tons prohibited		|
| 0.	      			| Speed limit (60km/h)					 		|
| 0.				    | Ahead only      								|


For the **image #3**, the model is relatively sure that this is a **Speed limit (20km/h) sign** (probability of 0.84033), and the image does contain a **Speed limit (20km/h) sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84033         		| Speed limit (20km/h)   						| 
| .14036     			| Speed limit (30km/h) 							|
| .00773				| Roundabout mandatory							|
| .00387      			| End of speed limit (80km/h)					|
| .00322			    | Speed limit (120km/h)      					|


For the **image #4**, the model is relatively sure that this is a **General caution sign** (probability of .8663), **but** the image does contain a **Traffic signals sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .8663         		| General caution   							| 
| .10245     			| Right-of-way at the next intersection 		|
| .01766				| Traffic signals								|
| .00839	      		| Pedestrians					 				|
| .0014				    | Beware of ice/snow      						|


For the **image #5**, the model is relatively sure that this is a **Stop sign** (probability of 1.0), and the image does contain a **Stop sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Stop   										| 
| 0.     				| Speed limit (30km/h) 							|
| 0.					| Speed limit (70km/h)							|
| 0.	      			| Turn right ahead						 		|
| 0.				    | No entry      								|


For the **image #6**, the model is relatively sure that this is a **Priority road sign** (probability of 1.0), and the image does contain a **Priority road sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Priority road   								| 
| 0.     				| Road work 									|
| 0.					| No vehicles									|
| 0.	      			| End of all speed and passing limits			|
| 0.				    | Speed limit (60km/h)      					|


For the **image #7**, the model is relatively sure that this is a **Yield sign** (probability of 0.99999), and the image does contain a **Yield sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99999     			| Yield   										| 
| 0.     				| Speed limit (50km/h) 							|
| 0.					| Speed limit (30km/h)							|
| 0.	      			| Speed limit (60km/h)					 		|
| 0.				    | No vehiclesy      							|


For the **image #8**, the model is relatively sure that this is a **Turn right ahead sign** (probability of 0.98381), and the image does contain a **Turn right ahead sign**. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98381         		| Turn right ahead   							| 
| .01457     			| Ahead only 									|
| .00119				| Road work										|
| .00014	      		| No passing for vehicles over 3.5 metric tons	|
| .00009			    | Go straight or left      						|


###Visualize the Neural Network's State with Test Images
Let's have a look for 3 examples:

Example 1. Convolution</b> (layer 1 after 'tf.nn.conv2d' operation)
It's possible to see shape of sign, but there are many background noizes.

Here is an output: feature maps of layer 1 convolution.

![alt text][image5]

Example 2. ReLU-activation (layer 1 after 'tf.nn.relu' operation)
Noize level (from backgroung objects) significantly reduced. This demonstrates that the CNN learned to detect useful traffic sign features on its own.

Here is an output: feature maps of layer 1 ReLU-activation.

![alt text][image6]

Example 3. Max-pooling</b> (layer 1 after 'tf.nn.max_pool' operation)
Image size reduced, but important features are available

Here is an output: feature maps of layer 1 max-pooling.

![alt text][image7]
