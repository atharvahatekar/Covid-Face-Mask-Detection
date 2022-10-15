# Covid-Face-Mask-Detection
Face Mask Detection System built by Keras/TensorFlow, Deep Learning and Computer Vision concepts
COVID Face Mask Detection

In this project I have used neural networks from deep learning to detect mask in a picture with the help of frameworks like tensorflow, keras, sklearn, matplotlib etc.
I have also took help from others and from internet to get the model working in a video stream by looping the code for a picture for each frame of the video.
Dataset:
I have used my custom dataset by collection pictures from the 
internet of masked and unmasked people and took pictures from 
some other small datasets. So, there are variety of pictures in the 
dataset.
I have used 3000 images of people with mask on them and also
3000 pictures of people without mask.


#With Mask:
<div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align = "center">
<img src="mask image.png" height="300" width="300" style="border-radius:3px;border:solid 1px #666" />
</p>
</div>

#without Mask:

<div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align = "center"
<img src="whithout mask image.png" height="300" width="300" style="border-radius:3px;border:solid 1px #666" />
</P>
</div>

#Training with the help of Tensorflow and Keras:
Now that we’re done with our face mask dataset, let’s use Keras 
and TensorFlow to train a classifier to automatically detect 
whether a person is wearing a mask or not.
To accomplish this task, we’ll be fine-tuning the MobileNet V2
architecture, a highly efficient architecture that can be applied to 
embedded devices with limited computational capacity.
Deploying our face mask detector to embedded devices could 
reduce the cost of manufacturing such face mask detection 
systems, hence why I choose to use this architecture.

#MobileNetV2:
It is a model developed by google.
MobileNetV2 is a convolutional neural network 
architecture that seeks to perform well on mobile devices. 
It is based on an inverted residual structure where the 
residual connections are between the bottleneck layers. 
The intermediate expansion layer uses lightweight 
depth wise convolutions to filter features as a source of 
non-linearity. As a whole, the architecture of 
MobileNetV2 contains the initial fully convolution layer 
with 32 filters, followed by 19 residual bottleneck layers.
I have completed training in jupyter notebook. In the project
there is a ‘train.ipynb’ file which contain code for training.

#SciKit Learn:
Scikit-learn (formerly scikits.learn and also known
as sklearn) is a free software machine learning library for the 
Python programming language.It features various classification, 
regression and clustering algorithms including support vector 
machines, random forests, gradient boosting, k-means and 
DBSCAN, and is designed to interoperate with the Python 
numerical and scientific libraries NumPy and SciPy.
Scikit-learn (sklearn) has been used for binarizing 
class labels, segmenting our dataset, and printing a classification 
report.

#Imutils:

It contains A series of convenience functions to make
basic image processing functions such as translation, rotation, 
resizing, skeletonization, and displaying Matplotlib images easier 
with OpenCV and both Python 2.7 and Python 3.
Imutils paths implementation will help us to find and list images
in our dataset. And we’ll use matplotlib to plot our training curve.


