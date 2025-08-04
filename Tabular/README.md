![](UTA-DataScience-Logo.png)

# Predicting Legendary Pokémon from Stats and Attributes

* This repository applies machine learning classification models to predict Legendary Pokémon using stats and metadata from the Pokémon dataset available on Kaggle:  [(provide link)](https://www.kaggle.com/datasets/shreyasdasari7/golden-pokdex-kanto-to-hoenn). 

## Overview

The challenge is to classify whether a Pokémon is Legendary or not based on its base stats (HP, Attack, Defense, Speed, etc.), typing, generation, and other features. Our approach involves formulating the problem as a binary classification task and applying models such as Logistic Regression, K-Nearest Neighbors, and Random Forest. After preprocessing and cleaning the data, we split it into training, validation, and test sets to train and evaluate each model. Our best model (Random Forest) achieved perfect accuracy on the test set, with Logistic Regression and KNN also showing strong performance.

## Summary of Workdone

* Data
  * Type: CSV file with Pokemon features and labels
  * Input: CSV file with numerical and categorical features (e.g., Total stats, Speed, Type, Capture Rate)
  * Output: Binary target (Is Legendary)
  * Size: ~800 data points
  * Split: 70% Training, 15% Validation, 15% Test
 
* Preprocessing / Clean up
  * Removed unnecessary columns (names, IDs, duplicated columns)
  * Handled missing values (none in our dataset)
  * One-hot encoded categorical variables (Types)
  * Scaled numerical features using StandardScaler

* Data Visualization
  * Histograms comparing feature distributions between legendary and non-legendary classes
  * Boxplots and violin plots for top numerical features
  * Feature correlation heatmap
  * Feature importance plot for Random Forest

* Problem Formulation
  * Input: Pokemon features (numerical stats and types)
  * Output: Binary classification (0 = Not Legendary, 1 = Legendary)

* Models Used
  * Logistic Regression
  * K-Nearest Neighbors (k=5)
  * Random Forest Classifier

* Hyperparameters
  * KNN: k=5
  * Random Forest: n_estimators=100, class_weight='balanced'
  * Logistic Regression: max_iter=1000

* Training
  * Environment: Python, Scikit-learn, Jupyter Notebooks
  * Training time: Less than a minute per model
  * Stopping criteria: Early validation performance and overfitting risk
  * No significant training difficulties

### Performance Comparison



### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







