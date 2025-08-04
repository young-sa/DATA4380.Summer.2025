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
<img width="485" height="117" alt="Screenshot 2025-08-04 at 11 05 32 AM" src="https://github.com/user-attachments/assets/ffd3e3fc-6942-4bfb-8c43-ca3f949edbcb" />


<img width="1033" height="598" alt="Screenshot 2025-08-04 at 11 13 08 AM" src="https://github.com/user-attachments/assets/c5d2673e-1d03-4926-a057-6239692b31cb" />


<img width="1012" height="823" alt="Screenshot 2025-08-04 at 11 18 28 AM" src="https://github.com/user-attachments/assets/6771d339-0ac6-47a6-86d1-3964c4a45351" />
<img width="1016" height="754" alt="Screenshot 2025-08-04 at 11 18 55 AM" src="https://github.com/user-attachments/assets/4d1fb286-0e15-42dd-8e8b-d6129a326b6d" />


<img width="1131" height="320" alt="Screenshot 2025-08-04 at 11 20 51 AM" src="https://github.com/user-attachments/assets/e2dcac47-6b97-43ff-b564-c930b908612f" />


### Conclusions

Random Forest was the best performer, achieving perfect classification. Logistic Regression also achieved perfect accuracy and recall in this dataset, however, not as well as Random Forest. KNN underperformed slightly on recall but was still fairly effective.

### Future Work

We can later use cross-validation and hyperparameter tuning (e.g., GridSearchCV), possibly explore ensemble methods, try dimensionality reduction (e.g., PCA). Additionally, we could expand the dataset or define popularity beyond "Is Legendary".

### Overview of files in repository

* Datasets from zip file:
  * Hoenn.csv
  * Johto.csv
  * Kanto.csv
* utils.py: various functions that are used in cleaning and visualizing data.
* YoungTabularFeasability.ipynb: cleans the dataset to prepare for machine learning.
* YoungTabularPrototype.ipynb: loads multiple trained models and compares results.

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







