import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from math import sqrt

# To load the dataset for the code to be used
data = pd.read_csv('C:\\Users\\User 1\\Desktop\\Assignment\\Degree\\Data Mining\\houses.csv')

# Function used to clean the price column
def clean_price(price):
    if isinstance(price, str):
        # Removes words such as RM, commas, spaces, and strip whitespace
        cleaned = price.replace('RM', '').replace(',', '').replace(' ', '').strip()
        return float(cleaned)
    return price

# Function to extract numeric values from string containing numbers
def extract_numeric(value):
    if pd.isna(value) or value == '-':
        return np.nan
    return pd.to_numeric(''.join(filter(str.isdigit, str(value))), errors='coerce')

# To help with cleaning data and preprocessing the data
def preprocess_data(df):
    # create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # to clean the price column
    df_clean['price'] = df_clean['price'].apply(clean_price)
    
    # to clean the numeric column
    numeric_columns = ['Bedroom', 'Bathroom', 'Property Size', 'Completion Year', 
                      '# of Floors', 'Total Units', 'Parking Lot', 'Firm Number']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(extract_numeric)
    
    # to create binary columns for facilities
    if 'Facilities' in df_clean.columns:
        common_facilities = ['Parking', 'Security', 'Swimming Pool', 'Playground', 
                           'Barbeque', 'Jogging Track']
        for facility in common_facilities:
            df_clean[f'has_{facility.lower().replace(" ", "_")}'] = df_clean['Facilities'].str.contains(
                facility, case=False, na=False).astype(int)
    
    # to create binary columns for nearby amenities
    nearby_columns = ['nearby school', 'nearby mall', 'Bus Stop', 'Mall', 
                     'Park', 'School', 'Hospital', 'Highway', 'Railway Station']
    for col in nearby_columns:
        if col in df_clean.columns:
            df_clean[f'has_{col.lower().replace(" ", "_")}'] = df_clean[col].notna().astype(int)
    
    # to help encode categorical columns
    categorical_columns = ['Category', 'Tenure Type', 'Property Type', 'Land Title', 'Floor Range']
    label_encoders = {}
    for col in categorical_columns:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].fillna('missing'))
            label_encoders[col] = le
    
    # Select features for modeling
    feature_columns = (
        # Numeric columns
        [col for col in numeric_columns if col in df_clean.columns] +
        # Encoded categorical columns
        [f'{col}_encoded' for col in categorical_columns if col in df_clean.columns] +
        # Binary facility columns
        [col for col in df_clean.columns if col.startswith('has_')]
    )
    
    # Create feature matrix
    X = df_clean[feature_columns]
    y = df_clean['price']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, feature_columns

def plot_correlation_matrix(X, y):
    """
    Create and plot a correlation matrix including target variable
    """
    # Combine features and target
    data_corr = X.copy()
    data_corr['price'] = y
    
    # Calculate correlation matrix
    correlation_matrix = data_corr.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(20, 16))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                linewidths=0.5)
    
    plt.title('Correlation Matrix of Features and Price')
    plt.tight_layout()
    plt.show()
    
    # Return highly correlated features with price
    price_correlations = correlation_matrix['price'].sort_values(ascending=False)
    print("\nTop 10 features most correlated with price:")
    print(price_correlations[1:11])  # Excluding price's correlation with itself

def perform_rfe(X, y, n_features_to_select=10):
    """
    Perform Recursive Feature Elimination
    """
    # Initialize the model to use for RFE
    base_model = RandomForestRegressor(random_state=123)
    
    # Initialize RFE
    rfe = RFE(estimator=base_model,
              n_features_to_select=n_features_to_select,
              step=1)
    
    # Fit RFE
    rfe = rfe.fit(X, y)
    
    # Get selected features
    selected_features = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    # Plot selected features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=selected_features[selected_features['Selected']],
                x='Rank',
                y='Feature')
    plt.title('Selected Features from RFE')
    plt.xlabel('Rank')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    return selected_features, rfe

def plot_model_metrics(metrics_df):
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(['MAE', 'RMSE', 'R-Squared', 'Adjusted R-Squared']):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=metrics_df.index, y=metrics_df[metric])
        plt.title(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance_dict):
    for name, importance_df in feature_importance_dict.items():
        if len(importance_df) > 0:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title(f'Top 10 Most Important Features - {name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

def plot_cv_results(cv_results_dict):
    """
    Plot cross-validation results comparison
    """
    plt.figure(figsize=(10, 6))
    models = list(cv_results_dict.keys())
    scores = [results['best_score'] for results in cv_results_dict.values()]
    
    sns.barplot(x=models, y=scores)
    plt.title('Best Cross-Validation RMSE by Model')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()

# Prepare the data
print("Preprocessing data...")
X, y, feature_columns = preprocess_data(data)

# Plot correlation matrix
print("\nGenerating Correlation Matrix...")
plot_correlation_matrix(X, y)

# Perform RFE
print("\nPerforming Recursive Feature Elimination...")
rfe_results, rfe_selector = perform_rfe(X, y)
print("\nRFE Selected Features:")
print(rfe_results[rfe_results['Selected']])

# Use RFE selected features
X_selected = X[rfe_results[rfe_results['Selected']]['Feature']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=123)

# Define parameter grids for each model
param_grids = {
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=123),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=123),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting (XGBoost)': {
        'model': XGBRegressor(random_state=123),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'Support Vector Regression (SVR)': {
        'model': SVR(),
        'params': {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.3]
        }
    }
}

def perform_grid_search(X_train, X_test, y_train, y_test, param_grids):
    """
    Perform grid search for each model and return best models and their scores
    """
    best_models = {}
    all_metrics = {}
    feature_importance_dict = {}
    cv_results_dict = {}
    
    for name, model_info in param_grids.items():
        print(f"\nPerforming Grid Search for {name}...")
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Store best model
        best_models[name] = grid_search.best_estimator_
        
        # Make predictions with best model
        y_pred = grid_search.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate Adjusted R-squared
        n = len(y_test)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Store metrics
        all_metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R-Squared': r2,
            'Adjusted R-Squared': adj_r2
        }
        
        # Store feature importance if available
        if hasattr(grid_search.best_estimator_, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': grid_search.best_estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance_dict[name] = feature_importance
        
        # Store CV results
        cv_results_dict[name] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_  # Convert back to positive RMSE
        }
        
        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)
        print(f"Best CV RMSE: {sqrt(-grid_search.best_score_):.2f}")
        print(f"Test Set Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R-Squared: {r2:.2f}")
        print(f"Adjusted R-Squared: {adj_r2:.2f}")
    
    return best_models, all_metrics, feature_importance_dict, cv_results_dict

# Perform grid search and get results
print("\nPerforming Grid Search for all models...")
best_models, all_metrics, feature_importance_dict, cv_results_dict = perform_grid_search(
    X_train, X_test, y_train, y_test, param_grids
)

# Plot results
print("\nPlotting model metrics...")
plot_model_metrics(pd.DataFrame(all_metrics).T)

# Plot cross-validation results
print("\nPlotting cross-validation results...")
plot_cv_results(cv_results_dict)

# Plot feature importance for applicable models
print("\nPlotting feature importance...")
plot_feature_importance(feature_importance_dict)