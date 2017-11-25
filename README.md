# CSE527 Final Project

###IMPORTANT:
The trained model cannot be uploaded into Github, you can download from [here](https://drive.google.com/a/cs.stonybrook.edu/file/d/1AcWFEFYmsfULmaVKdfd6M_mHbYD9_Ngn/view?usp=sharing), save the file in the same folder with the program, do not change its name.

The training and test data originally contains only 30images each which can be found [here](http://brainiac2.mit.edu/isbi_challenge/downloads). Since we are training deep model, those are far beyond enough. Hence, heavy augmentation is used. From initial 30 images, we augmented into 8820 images using multiple methods including rotation, shift, intensity changes, and elastic transformation. Details are in the section below.

In order to train the network, download these [augmented data](https://drive.google.com/drive/folders/1zikzGhtTe-RR-LzRBKXMx2D6vu2Ksy0i?usp=sharing). Put all .npy into folder data/dataset and run UnetTrain.py

# Quick Instruction to Run program
1. Clone all above files into local machine.
2. Download trained model as noted above from [here](https://drive.google.com/a/cs.stonybrook.edu/file/d/1AcWFEFYmsfULmaVKdfd6M_mHbYD9_Ngn/view?usp=sharing), save in the same folder as the file Run_Result.py
3. From terminal, run Run_Result.py with the format"python Run_Result.py [Input_image_path] [label_image]"

# ISBI Challenges: Segmentation of Neuronal Structures in EM stacks

This repository holds the code for the [ISBI Challenges](http://brainiac2.mit.edu/isbi_challenge/). It's meant to show how to construct Unets with Pytorch in a concise and straightforward way.

# Dependencies

 - [Pytorch 0.2.0](http://pytorch.org/)
 - Numpy
 - [OpenCV-Python](https://pypi.python.org/pypi/opencv-python)

# Implementation of deep learning framework -- Unet, using Pytorch

The architecture is originally from [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

[Provided data](http://brainiac2.mit.edu/isbi_challenge/) you can download the train and test data from this server.
you can also find data in the data folder.

### Pre-processing

The images are 3-D volume tiff, you should transfer the stacks into images first.
The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network.
To do data augumentation, an image deformation method was used, which was implemented in C++ using opencv.

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 10 epochs.

After 10 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy

---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0
* libtiff(optional)

Also, this code should be compatible with Python versions 2.7-3.5.

### Prepare the data

First transfer 3D volume tiff to 30 512*512 images.

To feed the unet, data augmentation is necessary.

An [image deformation](http://faculty.cs.tamu.edu/schaefer/research/mls.pdf) method is used, the code is

availabel in this [repository](https://github.com/cxcxcxcx/imgwarp-opencv).




### Define the model

* Check out ```get_unet()``` in ```unet.py``` to modify the model, optimizer and loss function.

### Train the model and generate masks for test images

* Run ```python unet.py``` to train the model.


After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight of your model's performance.

### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
