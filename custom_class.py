from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score
import numpy as np

class CustomCrossValidator:
    
    log = {}
    
    def __init__(self, model, X, y, sampling_method='none'):
        self.model = model
        self.X = X
        self.y = y
        self.sampling_method = sampling_method
        
    def _preprocess_data(self, X_train, X_val):
        # Perform OneHotEncoding on categorical columns in X_train and X_val
        categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            encoder = OneHotEncoder(sparse=False, drop='first')
            X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
            X_val_encoded = encoder.transform(X_val[categorical_columns])
            
            X_train = X_train.drop(columns=categorical_columns)
            X_val = X_val.drop(columns=categorical_columns)
            
            X_train = np.hstack((X_train_encoded, X_train.values))
            X_val = np.hstack((X_val_encoded, X_val.values))
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled
    
    def _apply_sampling(self, X_train, y_train):
        if self.sampling_method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif self.sampling_method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif self.sampling_method == 'smote':
            sampler = SMOTE(random_state=42)
        else:
            return X_train, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def _evaluate_model(self, model, X_train, y_train, X_val, y_val):
        # Clone the model to ensure a fresh copy for each fold
        temp_model = clone(model)
        
        # Apply sampling if specified
        X_train_resampled, y_train_resampled = self._apply_sampling(X_train, y_train)
        
        # Fit the model on the resampled training data
        temp_model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate the model on validation data using recall score
        y_pred = temp_model.predict(X_val)
        metrics = recall_score(y_val, y_pred)
        
        return metrics
    
    def cross_validate(self, folds=5):
        kfold = StratifiedKFold(n_splits=folds)
        scores = []
        
        for train_idx, val_idx in kfold.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            X_train_scaled, X_val_scaled = self._preprocess_data(X_train, X_val)
            
            metrics = self._evaluate_model(self.model, X_train_scaled, y_train, X_val_scaled, y_val)
            scores.append(metrics)
        
        mean_recall_score = np.mean(scores)
        model_name = self.model.__class__.__name__
        self.log[f'{model_name} {self.sampling_method}'] = mean_recall_score
        
        return scores