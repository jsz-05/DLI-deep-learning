## DLI-deep-learning:
This repository is a collection of interactive Python notebooks used in NVIDIA's Deep Learning course. 
[NVIDIA NGC](https://www.nvidia.com/en-us/gpu-cloud/) and [JupyterLab](https://jupyter.org/) were used for these projects.
---

## Index

### 1. Image classification with  [MNIST dataset using MLP](/01_mnist.ipynb)
MLP: Multi-Layer Perceptron
- (1) Data loading and visualization
- (2) Data editing (restructuring, normalization, categorical)
- (3) Model creation
- (4) Model compilation
- (5) Train the model on the data


### 2. Image classification of the  [American Sign Language (ASL) dataset using MLP](/02_asl.ipynb)

ASL: American Sign Language
- (1) Modelling a fully-connected neural network
- (2) Demonstrates High training accuracy
- (3) Demonstrates Low verification accuracy
- (4) Shows Examples of overfitting

### 3. Image classification of  [American Sign Language dataset using CNN](/03_asl_cnn.ipynb)

CNN: Convolutional Neural Networks
- (1) Analysis:
    - Increase verification accuracy with CNN
    - Training accuracy is still higher than validation accuracy
- (2) Solution:
    - Cleaned data provides better examples
    - Diversity in the dataset helps the generalization of the model

### 4. ASL image classification model improvement through [data augmentation](/04a_asl_augmentation.ipynb) & [model distribution and prediction](/04b_asl_predictions.ipynb)

#### A. [(Data Augmentation)](/04a_asl_augmentation.ipynb)

- Using `Keras` and [`ImageDataGenerator`](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)
  - `tensorflow.keras.preprocessing.image import ImageDataGenerator`
- Demonstrates Data Augmentation Techniques for increased identification accuracy
    - Image Flippimg
    - Rotation
    - Zooming
    - Move width and height
    - Homography
    - Brightness
    - Channel Shifting

#### B. [(Model Deployment)](/04b_asl_predictions.ipynb)

- (1) Load the trained model from disk
- (2) Changing the image format for models trained on images of different formats.
- (3) Perform inference with new images that the trained model encounters for the first time and evaluate performance

### 5. Using a Pre-trained model

#### A. Creating an [identification system for an automatic opening door](/05a_doggy_door.ipynb) using a pre-trained model ([VGG16](https://keras.io/api/applications/vgg/))

- (1) Load a pre-trained model (VGG16) using Keras
- (2) Preprocess your own images and work with a pre-trained model
- (3) Perform accurate inference on your own images using pre-trained models

#### B. Creating a customized opening g [automatic door for Bo through transfer learning](/05b_presidential_doggy_door.ipynb)

- Preparing a pre-trained model ([ImageNet Model](https://keras.io/api/applications/vgg/#vgg16-function)) for transfer learning
- Perform transfer learning on your own small dataset with pre-trained models
- Further fine-tuning the model for even better performance

### 6. Autocomplete text based on New York Times headlines (sequence data)

- (1) Preparing sequence data for use in RNN [(Recurrent Neural Network)](https://developers.google.com/machine-learning/glossary#recurrent-neural-network)
- (2) Building and training a model to perform word prediction

### 7. [Fresh and rotten fruit identification](/07_assessment.ipynb) (Final Project)

#
