import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

class CustomCrossValidator:
    
    log = {}
    
    def __init__(self, model, X, y, sampling_method='none'):
        self.model = model
        self.X = X
        self.y = y
        self.sampling_method = sampling_method
        self.trained_model = None
        self.best_model = None
        
    def preprocess_data(self, X_train, X_val):
        # Identify numeric and categorical features
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Scale numeric features
        scaler = StandardScaler()
        
        # Scale the train data
        X_train_scaled = scaler.fit_transform(X_train[numeric_features])
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index)
        
        #Scale the validation data
        X_val_scaled = scaler.transform(X_val[numeric_features])
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=numeric_features, index=X_val.index)
        
        # Encode categorical features
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Fit and transform on training data
        X_train_encoded = encoder.fit_transform(X_train[categorical_features])
        
        # Encode the validation dataset
        X_val_encoded = encoder.transform(X_val[categorical_features])
        
        # Extract feature names for encoded columns
        encoded_feature_names = []
        for feature, categories in zip(categorical_features, encoder.categories_):
            encoded_feature_names.extend([f"{feature}_{category}" for category in categories])
        
        # Create DataFrames with encoded information
        encoded_train_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
        encoded_val_df = pd.DataFrame(X_val_encoded, columns=encoded_feature_names, index=X_val.index)
        
        # Concat the DataFrames
        X_train_preprocessed = pd.concat([X_train_scaled_df, encoded_train_df], axis=1)
        X_val_preprocessed = pd.concat([X_val_scaled_df, encoded_val_df], axis=1)
        
        return X_train_preprocessed, X_val_preprocessed
        
    def apply_sampling(self, X_train, y_train):
        # Check which sampling method is specified and create the appropriate sampler
        if self.sampling_method == 'oversample':
            # Random oversampling
            sampler = RandomOverSampler(random_state=42)
        elif self.sampling_method == 'undersample':
            # Random undersampling
            sampler = RandomUnderSampler(random_state=42)
        elif self.sampling_method == 'smote':
            # Synthetic Minority Over-sampling Technique (SMOTE)
            sampler = SMOTE(random_state=42)
        else:
            # If no sampling method is specified, return the original data
            return X_train, y_train

        # Apply the sampling method to the training data
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        # Return the resampled training data
        return X_resampled, y_resampled
    
    def evaluate_model(self, X_train, y_train, X_val, y_val):
        # Clone the model to ensure a fresh copy for each fold
        temp_model = clone(self.model)
        
        # Apply sampling if specified
        X_train_resampled, y_train_resampled = self.apply_sampling(X_train, y_train)
        
        # Fit the model on the resampled training data
        temp_model.fit(X_train_resampled, y_train_resampled)
        
        # Store trained model
        self.trained_model = temp_model
        
        # Evaluate the model on validation data using recall score
        y_pred = temp_model.predict(X_val)
        metrics = recall_score(y_val, y_pred)
        
        return metrics
    
    def cross_validate(self, features=None, features_from=None, folds=5):
        kfold = StratifiedKFold(n_splits=folds)
        scores = []

        for train_idx, val_idx in kfold.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # Preprocess data
            X_train_preprocessed, X_val_preprocessed = self.preprocess_data(X_train, X_val)

            # Select specified features if provided, otherwise use all features
            if features is not None:
                X_train_preprocessed = X_train_preprocessed[features]
                X_val_preprocessed = X_val_preprocessed[features]

            # Evaluate the model
            metrics = self.evaluate_model(X_train_preprocessed, y_train, X_val_preprocessed, y_val)
            scores.append(metrics)

        mean_recall_score = np.mean(scores)
        model_name = self.model.__class__.__name__
        self.log[(model_name, features_from, self.sampling_method)] = mean_recall_score

        return scores
    
    def run_sampling_methods(self, sampling_methods, features=None, features_from=None, folds=5):
        # Iterate through each sampling method provided
        for method in sampling_methods:
            # Set the current sampling method
            self.sampling_method = method

            # Print the current sampling method to indicate progress
            print(f"Running cross-validation with sampling method: {method}")
            print()

            # Run cross-validation with the specified sampling method, features, and number of folds
            self.cross_validate(features=features, features_from=features_from, folds=folds)

        # Print the log of results after all sampling methods have been processed
        print("Log:", self.log)
        
    def run_grid_search(self, param_grid, scoring='recall', cv=5, features=None):
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=1, n_jobs=-1)

        # Initialize empty lists to store results
        scores = []
        best_models = []

        for train_idx, val_idx in grid_search.cv.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Preprocess data
            X_train_preprocessed, X_val_preprocessed = self.preprocess_data(X_train, X_val)
            
            # Apply sampling if specified (if needed)
            if self.sampling_method is not None:
                X_resampled, y_resampled = self.apply_sampling(X_train_preprocessed, y_train)
            else:
                X_resampled, y_resampled = X_train_preprocessed, y_train

            # Fit GridSearchCV on preprocessed data
            grid_search.fit(X_resampled, y_resampled)

            # Evaluate on validation set
            y_pred = grid_search.best_estimator_.predict(X_val_preprocessed)
            score = recall_score(y_val, y_pred)

            # Store results
            scores.append(score)
            best_models.append(grid_search.best_estimator_)

        # Print mean score
        mean_score = np.mean(scores)
        print(f'Mean Recall Score: {mean_score}')

        # Find best model based on validation scores
        best_model_idx = np.argmax(scores)
        self.best_model = best_models[best_model_idx]

        # Print best parameters and best score found during grid search
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        return grid_search