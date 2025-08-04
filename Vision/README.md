

# Weather Image Classification with Transfer Learning

* **One Sentence Summary** This repository presents a deep learning project using transfer learning models (MobileNetV2, ResNet50, DenseNet121) to classify weather conditions (Cloudy, Rain, Shine, Sunrise) from image data. [(provide link)](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset). 

## Overview

The goal of this project is to apply deep learning techniques for image classification of weather conditions using real-world weather imagery. We trained and compared three transfer learning models (MobileNetV2, ResNet50, and DenseNet121) to assess their effectiveness in classifying images into four weather-related classes.

Our best-performing model, MobileNetV2, achieved near-perfect AUC scores across all classes, demonstrating the capability of lightweight networks for high-quality image classification tasks.

## Summary of Workdone

* Organized and preprocessed a weather image dataset.
* Built a baseline CNN model.
* Applied data augmentation and evaluated its impact.
* Trained three transfer learning models: MobileNetV2, ResNet50, DenseNet121.
* Evaluated model performance using ROC curves and classification reports

### Data

* Type: JPEG/PNG weather images
* Input: Images labeled as Cloudy, Rain, Shine, Sunrise
* Size: Approximately ~1,100 images total
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
* Training Challenges
    * EfficientNetB0 was discarded due to shape mismatch issues
    * Handled grayscale images by converting to RGB using tf.image.grayscale_to_rgb()

### Performance Comparison

<img width="465" height="199" alt="Screenshot 2025-08-04 at 5 28 08 PM" src="https://github.com/user-attachments/assets/5572ce72-0fa1-40d5-8560-666953636294" />

<img width="834" height="815" alt="Screenshot 2025-08-04 at 5 03 29 PM" src="https://github.com/user-attachments/assets/34ded659-0eaa-47f8-9ee1-63786f883da2" />

<img width="1135" height="462" alt="Screenshot 2025-08-04 at 5 26 12 PM" src="https://github.com/user-attachments/assets/60428fb4-7829-4520-a72b-0a8d6a9208b4" />

<img width="1128" height="463" alt="Screenshot 2025-08-04 at 5 26 41 PM" src="https://github.com/user-attachments/assets/0adaa851-e4c3-49f0-8a55-1833ed7187f8" />

<img width="1122" height="461" alt="Screenshot 2025-08-04 at 5 27 08 PM" src="https://github.com/user-attachments/assets/61f2181d-f02f-4164-820d-52ad6d01767f" />

<img width="1119" height="450" alt="Screenshot 2025-08-04 at 5 30 55 PM" src="https://github.com/user-attachments/assets/17c9ea97-4fe5-46f4-a6b9-43fce7c6f58a" />

<img width="1126" height="455" alt="Screenshot 2025-08-04 at 5 31 17 PM" src="https://github.com/user-attachments/assets/760dae5d-44d9-4902-9425-d3af8f54cd89" />

<img width="1031" height="599" alt="Screenshot 2025-08-04 at 4 12 44 PM" src="https://github.com/user-attachments/assets/efc9558b-b7b1-4d07-bc03-79aa569e96a4" />


### Conclusions

MobileNetV2 achieved the best overall performance, with excellent balance across all classes. DenseNet121 was strong but dropped slightly on Class 2 and 3. ResNet50 underperformed compared to the other two models. It especially struggled on "Rain" and "Shine" classes.

### Future Work

We could expand the dataset to include more images, try other architectures like InceptionV3 or ConvNeXt, deploy model to a real-time weather monitoring application, and possibly fine-tune model weights instead of freezing base layers.

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







