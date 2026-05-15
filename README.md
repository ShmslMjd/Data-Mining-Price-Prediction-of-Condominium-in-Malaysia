# Data Mining: Price Prediction of Condominium in Malaysia

## Overview

This project focuses on data mining techniques to predict condominium prices in Malaysia. Using a comprehensive dataset of property listings, the project employs machine learning algorithms to analyze various features such as property size, location, facilities, and amenities to estimate property values accurately.

## Features

- **Data Preprocessing**: Comprehensive cleaning and transformation of raw property data
- **Feature Engineering**: Creation of binary features for facilities and amenities, categorical encoding
- **Exploratory Data Analysis**: Correlation analysis and visualization of feature relationships
- **Feature Selection**: Recursive Feature Elimination (RFE) for optimal feature subset selection
- **Machine Learning Models**: Implementation of multiple regression algorithms including:
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
  - Support Vector Regression (SVR)
- **Model Evaluation**: Comprehensive metrics including MAE, RMSE, R-Squared, and Adjusted R-Squared
- **Hyperparameter Tuning**: Grid search cross-validation for optimal model parameters
- **Visualization**: Correlation matrices, feature importance plots, and model performance comparisons

## Dataset

The project uses a CSV dataset (`houses.csv`) containing Malaysian condominium listings with the following key attributes:

- Property details (bedrooms, bathrooms, size, completion year)
- Location and proximity features (nearby schools, malls, transportation)
- Facilities and amenities
- Pricing information
- Tenure and property type classifications

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Installation

1. Clone or download the project repository
2. Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

## Usage

1. Ensure `houses.csv` is in the project directory
2. Run the main script:
   ```bash
   python DM-Python-Ver.py
   ```

The script will:
- Load and preprocess the data
- Generate correlation analysis visualizations
- Perform feature selection using RFE
- Train and evaluate multiple machine learning models
- Display performance metrics and feature importance plots

## Methodology

### Data Preprocessing
- Price column cleaning (removing currency symbols and formatting)
- Numeric feature extraction from mixed data types
- Binary encoding of categorical facilities and amenities
- Label encoding of categorical variables
- Missing value imputation and feature scaling

### Feature Selection
- Recursive Feature Elimination (RFE) using Random Forest as base estimator
- Selection of top 10 most relevant features for modeling

### Model Training
- Train-test split (80-20)
- Grid search hyperparameter optimization with 5-fold cross-validation
- Evaluation on test set with multiple performance metrics

## Results

The project compares four machine learning models for price prediction:
- Decision Tree
- Random Forest
- XGBoost (Gradient Boosting)
- Support Vector Regression

Performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-Squared
- Adjusted R-Squared