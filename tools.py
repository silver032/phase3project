import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.30, random_state=42):
    """
    Split the DataFrame into training and testing sets based on the target variable.
    ---
    Inputs:
        df (DataFrame): The input DataFrame containing both features and the target variable.
        target (str): The name of the target variable column.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.30.
        random_state (int, optional): Controls the shuffling applied to the data before splitting. Default is 42.
    ---
    Outputs:
        X_train, X_test, y_train, y_test: The train-test split arrays of features and target variable.
    """
    X = df.drop([target], axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_numeric_features(X_train, X_test):
    """
    Scale numeric features using StandardScaler.
    ---
    Inputs:
        X_train (DataFrame): Training data features.
        X_test (DataFrame): Testing data features.
    ---
    Outputs:
        X_train_scaled_df, X_test_scaled_df (DataFrame): Scaled numeric feature DataFrames for training and testing data.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    
    # Fit and transform on training data
    X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_features])
    X_train_numeric_scaled_df = pd.DataFrame(X_train_numeric_scaled, columns=numeric_features, index=X_train.index)
    
    # Transform on test data
    X_test_numeric_scaled = scaler.transform(X_test[numeric_features])
    X_test_numeric_scaled_df = pd.DataFrame(X_test_numeric_scaled, columns=numeric_features, index=X_test.index)
    
    return X_train_numeric_scaled_df, X_test_numeric_scaled_df

def encode_categorical_features(X_train, X_test):
    """
    Encode categorical features using OneHotEncoder.
    ---
    Inputs:
        X_train (DataFrame): Training data features.
        X_test (DataFrame): Testing data features.
    ---
    Outputs:
        encoded_train_df, encoded_test_df (DataFrame): Encoded categorical feature DataFrames for training and testing data.
    """
    # Identify categorical columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Fit and transform on training data
    encoded_train = encoder.fit_transform(X_train[categorical_features])
    
    # Transform on test/validation data
    encoded_test = encoder.transform(X_test[categorical_features])
    
    # Extract feature names for encoded columns
    encoded_feature_names = []
    for feature, categories in zip(categorical_features, encoder.categories_):
        encoded_feature_names.extend([f"{feature}_{category}" for category in categories])
    
    # Create DataFrames with encoded information
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_feature_names, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=X_test.index)
    
    
    return encoded_train_df, encoded_test_df

def combine_features(X_numeric, X_categorical):
    """
    Combine scaled numeric and encoded categorical features into a single DataFrame.

    Inputs:
        X_numeric (DataFrame): Scaled numeric features DataFrame.
        X_categorical (DataFrame): Encoded categorical features DataFrame.

    Output:
        X_preprocessed (DataFrame): Combined DataFrame containing both numeric and categorical features.
    """
    X_preprocessed = pd.concat([X_numeric, X_categorical], axis=1)
    return X_preprocessed