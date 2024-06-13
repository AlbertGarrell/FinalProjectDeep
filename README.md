# Melanoma Cancer Prediction using CNNs and Metadata

Due to the size of the files and data, we couldn't upload them directly to GitHub. Here is the link to the full project:

**Link to the project:** [Final Project](https://drive.google.com/drive/folders/15N_DElOU514RqPHLUr701LCyyzzHT_hy?usp=drive_link)

## Authors

- Maren Clapers Colet - NIA 243397
- Albert Garrell Golobardes - NIA 254635
- Clara Jaurés Bancells - NIA 254177

## Contents

1. [Introduction to the Project](#1-introduction-to-the-project)
2. [Original Dataset](#2-original-dataset)
3. [Expanded Dataset](#3-expanded-dataset)
4. [Evolution of the Tests](#4-evolution-of-the-tests)
5. [Best Base Models](#5-best-base-models)
6. [GRAD-CAM and Practical Use](#6-grad-cam-and-practical-use)
7. [Conclusion](#7-conclusion)
8. [Future Work](#8-future-work)

## 1. Introduction to the Project

### Motivation and Objectives
- **Problem:** Melanoma accounts for 75% of deaths due to skin cancer.
- **Objective:** Assist dermatologists in accurately identifying melanoma lesions.
- **Data Source:** Kaggle competition organized by SIIM-ISIC, providing images of moles and contextual patient information (https://www.kaggle.com/c/siim-isic-melanoma-classification/overview).

## 2. Original Dataset

### Initial Characteristics
- **Total Images:** 33.1K images in 7 different sizes.
- **Class Distribution:** 98.23% benign and 1.77% malignant.
- **Metadata:** Includes fields such as image name, patient ID, sex, approximate age, anatomical site, diagnosis, and more.
- **Link:** https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092

### Data Cleaning Steps
1. Removed irrelevant columns (tfrecord, width, height).
2. Imputed missing age values with the mean.
3. Removed records with missing sex.
4. Filled missing anatomical site values with 'unknown'.
5. Normalized the age values.
6. Encoded categorical variables, such as sex and anatomical site.

**Final Dataset Size:** 33,061 records with 12 columns, from which 9 columns were used as metadata to train the classifier of the models (excluding image_name, patient_id, and the target).

## 3. Expanded Dataset

### Characteristics
- **Total Images:** 60.4K images.
- **Objective:** Increase the number of malignant cases (target = 1) compared to the original dataset.
- **Class Distribution:** 91.26% benign and 8.74% malignant.
- **Data Cleaning:** Similar to the original dataset, but with the addition of sex imputation using Machine Learning.
- **Link:** https://www.kaggle.com/datasets/shonenkov/melanoma-merged-external-data-512x512-jpeg

**Final Dataset Size:** 31,413 records with 13 columns, from which 10 columns were used as metadata to train the classifier of the models (excluding image_name, patient_id, and the target).

### Sex Imputation with Machine Learning
- Used a Decision Tree Classifier to infer missing sex values based on age and the anatomical site of the mole.

### Increasing Malignant Cases
- Randomly removed 27,000 benign records to balance the dataset.
- Following this change, the proportion of malignant images increased from 8.74% to 16.25%.

## 4. Evolution of the Tests

In order to evaluate the models, we were extracting some metrics both in training (and validation) and testing. The metrics used were the following: Accuracy, Recall, Precision, F1 score, ROC AUC score and the Confusion Matrix.

**For each model, there is the model trained in the Results folder, and in the Models_txt, there is the model class defined with the name of the model itself. Also, either in the document Arq_ModelsProves_DataSet1.docx or Arq_ModelsProves_DataSet2.docx (depending on the dataset used), there is documented all the specs and details of each model, and the plots of the metrics evaluating the training and evaluation part for every epoch, and the results for all metrics for the testing part and the given confusion matrix.**

In order to guide the evolution process of our models, we mainly evaluated how good or bad was a model based on two metrics:
- **Recall** (ratio of true positives to the sum of true positives and false negatives), as we are dealing with a medical classifying problem, and the main goal of the models created is to find and classify as positive as many positives as possible, as we don't want a patient with a malignant mole, to be classified with our model as negative, as the repercussions of an error in such cases can have fatal consequences.
- **Precision** (ratio of true positives to the sum of true positives and false positives), because we also need to take into account all the patients with benign moles that initially are being classified as malignant. 

### Initial Experiments
- **Baseline Models:** Initially tested various pretrained models like MobileNetV2, ResNet, and EfficientNet on the original dataset. We finally opted for ResNet50, given the computational capabilities that we had, it was the one that provided a better ration between results, time spent training and capabilities of the GPU. We decided to use ResNet50, as it is a model suitable for large-scale datasets and applications where higher accuracy is needed without excessive computational cost.
- **Image Sizes:** We tested with different image sizes, as we want to use higher quality images (1024x1024), but we didn't have the resources and the computational capabilities to execute a model with such image sizes, so finally we used in the original dataset 256x256 image sizes, rescaled to 224x224 to adjust the size to the input of the pretrained models. for the second dataset, the images were 512x512, but again we resized to 224x224.
- **Challenges:** Encountered class imbalance issues, leading to poor recall for malignant cases.

### Stratified Sampling and Data Augmentation
- **Stratified Sampling:** Introduced stratified sampling to ensure a higher proportion of malignant cases in the training set, as we had a really disproportionate datasets and the models weren't learning to identify te¡he positive moles.
- **Data Augmentation:** We applied different transformations, in order to generate models invariants to certain transformations, and to compensate the stratified sampling, as we heavily increased the proportion of the 2 classes in the training set, and as a consequence, a lot of positive images were repteated, generating a lot of overfitting which we tried to solve with data augmentation. Applied techniques such as horizontal flipping, brightness adjustment, Gaussian blur, and noise addition to increase the diversity of the training data, as well as resizing the inputs and normalizing the values.

### Weighted Loss Functions
- **Binary Cross-Entropy:** Used weighted binary cross-entropy to handle class imbalance more effectively by giving more importance to the malignant class.

### Expanded Dataset Experiments
- **Expanded Dataset:** Used a new dataset with a more balanced distribution of benign and malignant cases.
- **Improved Sampling:** Further refined stratified sampling techniques to maintain a realistic yet balanced representation of classes.
- **Enhanced Augmentation:** Continued to use data augmentation to prevent overfitting and improve model generalization.

### Model Training and Hyperparameter Tuning
- **Model Architectures:** Experimented with various architectures and configurations, testing with different number of FC layers, as well as testing with different reduction scaling of the features, and also testing an architecture where some FC layers were applied to the metadata before concatenating it to the output of the pretrained model, applying a classifier for both output of the pretrained model and the preprocessed metadata. (All the architectures can be found in the ArquitecturesClassifiers.docx file)
- **Optimizer and Learning Rate:** Tested different optimizers like Adam or SGD with various learning rates to find the most effective training configuration.
- **Regularization:** Incorporated dropout and batch normalization to prevent overfitting.

## 5. Best Base Models

Once we tested numerous models and refined those with the most promise, we concluded that there wasn't a single model that stood out as the best. Instead, we identified two base models, each with their own strengths. These models complement each other: where one falls short, the other works correctly. Therefore, using both models simultaneously with the same data to cross-validate the results should yield the most accurate final prediction.

Here is a brief description of both models, their strengths, and how they can be combined for optimal melanoma detection:

### Model 14 (Extended Dataset): High Recall

Model 14 focuses on high recall, ensuring most melanoma cases are detected. This model is ideal for initial screening, where missing a melanoma case is more critical than flagging a benign case.

**Model Definition and Metrics:** 
- Stratified Sampling: 25% malignant
- Adam optimizer with a learning rate (LR) of 0.0001
- Basic Data Augmentation
- Weighted Binary Cross-Entropy Loss with pos_weight=2.0
- **Metrics on Test:** 
  - Accuracy: 0.8028
  - **Recall: 0.8266**
  - **Precision: 0.4492**
  - F1: 0.5821
  - AUC: 0.8124
  - Confusion Matrix: [[4182 1058] [ 181  863]]
- The plots evaluating the training and validation, as well as the confusion matrix of the test, can be found in the Arq_ModelsProves_DataSet2.docx on pages 25 and 26.

**Strengths of Model 14:**
- **High Recall:** Ensures that nearly all true positive cases are detected.
- **Safety Net:** Acts as a safety measure to catch all potential melanoma cases, even if some are false positives.

### Model 20 (Extended Dataset): Balanced Model

Model 20 strikes a balance between recall and precision. It not only identifies a high number of positive cases but also maintains a relatively low number of false positives. This balance makes this model reliable for confirming true positive cases and reducing the number of benign cases mistakenly identified as malignant.

**Model Definition and Metrics:** 
- Stratified Sampling: 25% malignant
- Adam optimizer with LR=0.0001
- Basic Data Augmentation
- Weighted Binary Cross-Entropy Loss with pos_weight=2.0
- **Metrics on Test:** 
  - Accuracy: 0.8520
  - **Recall: 0.6686**
  - **Precision: 0.5445**
  - F1: 0.6002
  - AUC: 0.7786
  - Confusion Matrix: [[4656  584] [ 346  698]]
- The plots evaluating the training and validation, as well as the confusion matrix of the test, can be found in the Arq_ModelsProves_DataSet2.docx on pages 36 and 37.

**Strengths of Model 20:**
- **Balanced Metrics:** Provides a good trade-off between recall and precision, reducing false positives.
- **Confirmation:** Can be used to validate the positive cases identified by Model 14.

### Combining Model 14 and Model 20

To leverage the strengths of both models, they can be used in a complementary manner to improve overall diagnostic accuracy:

1. **Initial Screening with Model 14:**
   - Apply Model 14 for the initial screening of melanoma. This model will flag all potential cases with high sensitivity, ensuring that no possible melanoma case is missed. The priority at this stage is to maximize recall, even if it means a higher number of false positives.

2. **Validation with Model 20:**
   - The cases flagged by Model 14 can then be passed through Model 20 for validation. Model 20 will help filter out false positives, confirming true positive cases with higher precision. This two-step process ensures that only the cases most likely to be malignant are identified, reducing the burden of false positives on dermatologists.

By combining the models in this way, the system ensures comprehensive initial coverage (catching all potential melanoma cases) and then refines the results to reduce false positives. This approach maximizes both safety and efficiency, providing a robust tool for melanoma detection.

## 6. GRAD-CAM and Practical Use

### Gradient-weighted Class Activation Mapping (GRAD-CAM)
- **Visualization:** GRAD-CAM provides a way to visualize areas of an image that contribute most to the model's prediction. It produces heatmaps that highlight important regions in the image which the model focuses on during classification.

### Practical Application
- **Medical Report Generation:** For images detected as positive (potential melanoma), the practical use of GRAD-CAM can be extended to generate a comprehensive medical report. This report will include:
  - **Original Image:** The original, unaltered image of the lesion.
  - **GRAD-CAM Processed Image:** A heatmap overlay on the original image showing the areas of focus.
  - **Probability Score:** The probability assigned by the model indicating the likelihood of the lesion being malignant.
  - **Patient Information:** Relevant details such as patient ID, age, sex, and anatomical site of the mole.
  - **Anatomical Site:** The location of the mole on the patient's body.
  
- **Support for Dermatologists:** This practical application helps dermatologists detect possible malignant moles by highlighting the areas the model used to make its prediction. However, it is important to emphasize that the final diagnosis should always be made by a medical professional.

## 7. Conclusion

This project demonstrates the potential of deep learning in melanoma detection. By integrating convolutional neural networks (CNNs) with patient metadata, we improved classification accuracy, highlighting the value of multi-modal data.

Our best models, Model 14 and Model 20, balance recall and precision effectively. Using these models together allows for comprehensive screening and precise validation, reducing false negatives and positives. GRAD-CAM further aids dermatologists by highlighting key image areas relevant to the model's predictions.

While our models support melanoma detection, they are tools to assist, not replace, professional medical judgment. A qualified dermatologist should always make the final diagnosis.

This project lays the groundwork for future research. Advancements could include using more complex models, higher-resolution images, and genetic algorithms for optimization. As computational resources and methodologies improve, deep learning's potential in medical diagnostics will continue to grow.

## 8. Future Work

- **Advanced Models and Higher-Resolution:** With more resources and computational capabilities, we could test more complex models and use higher-resolution images to potentially improve the model’s performance. Exploring other pretrained models might also provide better fits for the dataset.
- **Increased Data Transformations:** Increased computational resources would enable the application of a broader range of data augmentations. This would help the model gain invariance to multiple aspects and reduce overfitting.
- **Genetic Algorithms for Model Optimization:** Employing genetic algorithms could be a powerful method for optimizing model selection and configuration. This method could systematically explore the search space and evolve models towards better performance, rather than relying on trial and error.
