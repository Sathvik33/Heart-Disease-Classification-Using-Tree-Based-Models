# Heart Disease Prediction with Decision Tree and Random Forest

This project demonstrates the application of **Decision Tree** and **Random Forest** algorithms to predict the likelihood of heart disease based on several clinical features. The project includes data preprocessing, model training, evaluation, overfitting analysis, and model visualization.

## 📝 Project Overview

The goal of this project is to predict whether a patient has heart disease using their clinical data. The dataset includes features such as age, cholesterol level, blood pressure, and other clinical parameters. Two machine learning models are explored:

1. **Decision Tree Classifier**: A simple and interpretable model used to classify whether a patient has heart disease.
2. **Random Forest Classifier**: An ensemble method that combines multiple decision trees to improve prediction accuracy.

### Key Steps:
- **Data Preprocessing**: Label encoding for categorical variables and feature scaling using `MinMaxScaler`.
- **Model Training**: Trains Decision Tree and Random Forest models.
- **Overfitting Analysis**: Analyzes how model accuracy changes with tree depth to detect overfitting.
- **Model Evaluation**: Accuracy, Confusion Matrix, and Classification Report for both models.
- **Visualization**: Decision Tree visualization and feature importance from Random Forest.

## 🔍 Dataset

The dataset used in this project is the **Heart Disease Dataset**, which contains the following features:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- Oldpeak (Depression Induced by Exercise Relative to Rest)
- Slope of the Peak Exercise ST Segment
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia

The target variable is `target`, indicating the presence (1) or absence (0) of heart disease.

## 🛠️ Libraries Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Graphviz
