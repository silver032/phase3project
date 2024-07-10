import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.30, random_state=42):
    X = df.drop([target], axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_numeric_features(X_train, X_test):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_numeric_scaled = scaler.transform(X_test[numeric_features])
    return X_train_numeric_scaled, X_test_numeric_scaled, numeric_features

def encode_categorical_features(X_train, X_test):
    categorical_features = X_train.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    X_train_categorical_encoded = encoder.fit_transform(X_train[categorical_features])
    X_test_categorical_encoded = encoder.transform(X_test[categorical_features])

    encoded_feature_names = []
    for feature in categorical_features:
        categories = encoder.categories_[categorical_features.get_loc(feature)]
        encoded_feature_names.extend([f"{feature}_{category}" for category in categories[1:]])

    return X_train_categorical_encoded, X_test_categorical_encoded, encoded_feature_names

def combine_features(X_numeric, X_categorical, numeric_features, encoded_feature_names):
    X_preprocessed = np.hstack((X_numeric, X_categorical))
    return pd.DataFrame(X_preprocessed, columns=list(numeric_features) + encoded_feature_names)

