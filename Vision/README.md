

# Weather Image Classification with Transfer Learning

* **One Sentence Summary** This repository presents a deep learning project using transfer learning models (MobileNetV2, ResNet50, DenseNet121) to classify weather conditions (Cloudy, Rain, Shine, Sunrise) from image data. [(provide link)](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset). 

## Overview

The task is to classify weather conditions from static images into one of four categories: Cloudy, Rain, Shine, or Sunrise. This is a computer vision classification problem. We use transfer learning with pre-trained convolutional neural networks (CNNs) from Keras Applications. Our approach includes data augmentation, baseline training, and evaluation via ROC curves and classification reports. Among the three models, DenseNet121 showed the most consistent and high performance across all classes.

## Summary of Workdone

* Organized and preprocessed a weather image dataset.
* Built a baseline CNN model.
* Applied data augmentation and evaluated its impact.
* Trained three transfer learning models: MobileNetV2, ResNet50, DenseNet121.
* Evaluated model performance using ROC curves and classification reports

### Data

* Type: JPEG/PNG weather images
* Input: Images labeled as Cloudy, Rain, Shine, Sunrise
* Size: Approximately ~1,200 images total
* Instances: 80% training, 20% validation split
  
#### Preprocessing / Clean up

* Images resized to 180x180.
* Grayscale images converted to RGB.
* Applied rescaling, normalization, and data augmentation (rotation, zoom, flip).

#### Data Visualization

Sample images from each class were visualized.
* Rain images showed more motion blur or droplets.
* Shine/Sunrise had intense lighting conditions.
* Classes were reasonably balanced.

### Problem Formulation

* Input: RGB images of shape (180, 180, 3)
* Output: One of four weather labels (multi-class classification)
  * Models
    * MobileNetV2: Lightweight, fast, good performance.
    * ResNet50: Deeper network, but performance was inconsistent.
    * DenseNet121: Best performance across all classes.
  *Hyperparameters
    * Loss: Categorical Crossentropy
    * Optimizer: Adam
    * Learning rate: 0.0001
    * Batch size: 32
    * Epochs: 10 (early stopping based on validation loss)

### Training

* Software: Python, TensorFlow, Keras, scikit-learn
* Hardware: MacBook Pro (M1 chip), Google Colab (TPU/GPU optional)
* Training took 5–15 minutes depending on the model.
* Models saved using model.save()
* Training stopped early when validation loss plateaued.

### Performance Comparison



### Conclusions

DenseNet121 outperformed the other two models in both ROC and classification accuracy. MobileNetV2 was a close second, making it ideal for lightweight applications. ResNet50 struggled especially on "Rain" and "Shine" classes.

### Future Work

We could expand the dataset to include more images, try other architectures like InceptionV3 or ConvNeXt, deploy model to a real-time weather monitoring application, and possibly fine-tune model weights instead of freezing base layers.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* TrainBaseModel.ipynb: Trains a basic CNN model.
* TrainBaseModelAugmentation.ipynb: Adds augmentation to the baseline.
* Train-MobileNetV2.ipynb: Implements MobileNetV2 training.
* Train-ResNet50.ipynb: Implements ResNet50 training.
* Train-DenseNet121.ipynb: Implements DenseNet121 training.
* CompareModels.ipynb: Loads and compares ROC curves of the 3 models.

### Software Setup
* TensorFlow
* Keras
* NumPy
* Matplotlib
* scikit-learn

#### Performance Evaluation

* Run CompareModels.ipynb to load all saved models and generate ROC curves for validation set.


## Citations

* Keras Applications: https://keras.io/examples/vision/image_classification_from_scratch/
* Weather Dataset: https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset
* Chatgpt did help with solving errors that occured.







