# Melanoma Cancer Prediction using CNNs and Metadata

## Overview

This project aims to predict melanoma cancer by combining convolutional neural networks (CNNs) with patient metadata. The work is part of our Deep Learning course in the Bachelor's degree in Mathematical Engineering in Data Science, leveraging the dataset provided by the SIIM-ISIC Melanoma Classification competition on Kaggle.

## Contents

1. Introduction to the Project
2. Original Dataset
3. Expanded Dataset
4. Best Models
5. GRAD-CAM and Practical Use

## 1. Introduction to the Project

### Motivation and Objectives
- **Problem:** Melanoma accounts for 75% of deaths due to skin cancer.
- **Objective:** Assist dermatologists in accurately identifying melanoma lesions.
- **Data Source:** Kaggle competition organized by SIIM-ISIC, providing images of moles with contextual patient information.

## 2. Original Dataset

### Initial Characteristics
- **Total Images:** 33.1K images in 7 different sizes.
- **Class Distribution:** 98.23% benign and 1.77% malignant.
- **Metadata:** Includes fields such as image name, patient ID, sex, approximate age, anatomical site, diagnosis, and more.

### Data Cleaning Steps
1. Removed irrelevant columns (tfrecord, width, height).
2. Imputed missing age values with the mean.
3. Removed records with missing sex.
4. Filled missing anatomical site values with 'unknown'.
5. Normalized age values.
6. Encoded categorical variables (sex and anatomical site).

**Final Dataset Size:** 33,061 records with 12 columns.

## 3. Expanded Dataset

### Characteristics
- **Total Images:** 60.4K images.
- **Objective:** Increase the number of malignant cases (target = 1).
- **Class Distribution:** 91.26% benign and 8.74% malignant.
- **Data Cleaning:** Similar to the original dataset.

**Final Dataset Size:** 31,413 records with 13 columns.

### Sex Imputation with Machine Learning
- Used a Decision Tree Classifier to infer missing sex values based on age and anatomical site.

### Increasing Malignant Cases
- Randomly removed 27,000 benign records to balance the dataset.

## 4. Best Models

### Pretrained Models Tested
- **Efficient Net**
- **Mobile Net Version 2**
- **ResNet 50**

### Model Training
- **Stratified Sampling:** Increased the proportion of positive cases in the training set.
- **Data Augmentation:** Applied transformations like horizontal flip, brightness contrast, Gaussian blur, and Gaussian noise.

### Architecture and Hyperparameters
- **Base Model:** Fully connected layers with LeakyReLU activation.
- **Loss Function:** Binary Cross Entropy with weighted loss for class imbalance.
- **Optimizer:** Adam optimizer with varying learning rates.
- **Regularization:** Dropout and Batch Normalization.

### Evaluation Metrics
- **Model 14:** 
  - Stratified Sampling: 50% malignant
  - Adam optimizer with LR=0.0001
  - Simple Data Augmentation
  - Binary Cross Entropy Loss
  - **Metrics:** Accuracy: 0.8028, Recall: 0.8266, Precision: 0.4492, F1: 0.5821, AUC: 0.8124

- **Model 20:** 
  - Stratified Sampling: 25% malignant
  - Adam optimizer with LR=0.0001
  - Simple Data Augmentation
  - Weighted BCE Loss with pos_weight=2.0
  - **Metrics:** Accuracy: 0.8520, Recall: 0.6686, Precision: 0.5445, F1: 0.6002, AUC: 0.7786

## 5. GRAD-CAM and Practical Use

### Gradient-weighted Class Activation Mapping (GRAD-CAM)
- Visualizes areas of an image that contribute most to the model's prediction.
- Produces heatmaps highlighting important regions for diagnosis.

### Practical Application
- Provides valuable insights into the likelihood of a lesion being malignant.
- Assists dermatologists in making informed decisions based on model predictions.

## Conclusion

This project demonstrates the potential of deep learning in medical diagnostics. By integrating CNNs with metadata, we achieved significant accuracy in melanoma prediction, underlining the importance of multi-modal data in complex classification tasks.

## Future Work

- **Model Enhancement:** Experimenting with advanced architectures like EfficientNet or ResNet.
- **Metadata Enrichment:** Incorporating additional patient information to improve model performance.
- **Explainability:** Implementing more interpretability techniques to provide better insights into model decisions.
