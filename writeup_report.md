# **Behavioral Cloning**

## Write-up Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/forward.gif "Image in forward lap"
[image3]: ./examples/recovery_left.gif "Recovery from left"
[image4]: ./examples/recovery_right.gif "Recovery from right"
[image5]: ./examples/reverse.gif "Image in reverse lap"
[image6]: ./examples/track2.gif "Image in forward lap of track two"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 142-148)

The model includes RELU layers to introduce nonlinearity (code line 142-146), and the data is normalized in the model using a Keras lambda layer (code line 138).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 143, 145, 155, 157, 159 and 161).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 106-114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 164).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and right lane driving on track two.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to repurpose one of the existing models.

My first step was to use a convolution neural network model similar to the network used by Nvidia's DriveAV team as discussed in the lesson.

To combat the overfitting, I modified the model to include additional fully connected layers and added drop out layers (with 50% probability) after first 2 convolution layers and also after each fully connected layers before output layer.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 135-171) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture generated using plot_model from keras.utils

![Model Architecture][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Forward lap][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it goes off the track. These images show what a recovery looks like starting from left and right side of the road :

![Recovering from left][image3]
![Recovering from right][image4]

Then I collected the data by making a U turn of the car and driving for a full lap. Here is an example image of center lane driving in reverse direction :

![Reverse lap][image5]

Then I recorded a lap on track two using right lane driving. Here is an example image.

![Track2][image6]

I also made use of the images from left and right camera, initially I planned to construct a panaromic image by stitching the left, center and right images as single input to the network however this would have required changes to drive.py to be able to stitch the images before feeding to the network. To avoid the changes to drive.py I simply augmented the data by treating the left and right images as if they are center images and "guesstimated" the steering angle by applying an "adjustment factor" of 0.2 This number may not be optimal as I did not do experiments to arrive at best value.

To augment the data set further, I also flipped images and angles thinking that this would help model to learn driving under different condition and sort of normalize the steering angles in the training data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

After the collection process, I had 45054 training data points and 11268 validation data points. I then preprocessed this data by normalizing the images by dividing the samples by 255 and subtracting 0.5 (so that mean is 0) and cropping the images to remove the 70 pixels from the top (to hide the distracting information like trees/shadows/sun etc) and 25 pixels from the bottom (to hide the image of the car itself).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was choosen to be 10 as a sort of trade off between the training time and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Possible improvements

Though model runs fine for track one, it does not work well for track two. Following can improve the performance on track two.
I did not get a chance to try these out due to time limitation.

1. Train with more training data from track two. The training data from track one shows the center lane driving and not so sharp turns. Track two needs the right lane driving and has a lot of sharp turns.
2. Tune the model architecture (change the number of filters for convolution layers and play around the drop out probability rates)
3. Try changing the input to model from single center image to panaromic image obtained by stitching the left, center and right images.
