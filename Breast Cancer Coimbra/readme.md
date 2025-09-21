# Coimbra Breast Cancer Prediction Project

## Overview
This project implements a machine learning pipeline to predict breast cancer risk using the Coimbra dataset, which contains various biomedical features. The notebook covers data preprocessing, model training using both Support Vector Machines (SVM) and a Neural Network, and evaluation with medical-grade metrics.

## Dataset
- **Source**: Coimbra Breast Cancer Dataset (`dataR2.csv`)
- **Features**: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1
- **Target**: `Classification` (0 = No Cancer, 1 = Cancer)
- The dataset is balanced using oversampling techniques to improve model training.

## Project Structure
1. **Data Loading and Preprocessing**
   - Load the dataset.
   - Normalize features using Min-Max scaling.
   - Separate features and target variable.
   - Split data into training and testing subsets with stratification.

2. **Model Training**
   - Train a Support Vector Machine (SVM) with polynomial and RBF kernels.
   - Define and train a Neural Network with Keras.
   - Use callbacks to save best model checkpoints.

3. **Model Evaluation**
   - Calculate metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC.
   - Generate confusion matrices and classification reports.
   - Evaluate both training and testing datasets.

4. **Model Export**
   - Save the best Neural Network model.
   - Save reference columns, target name, scaling information, and the dataset for deployment.

## Usage
- Run the notebook step-by-step to train models and evaluate performance.
- Exported models and metadata can be integrated into deployment pipelines or dashboards.

## Requirements
- Python 3.x
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy



