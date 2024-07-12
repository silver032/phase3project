## Target Variable Investigation

There is a class imbalance between the three classes inside of the target variable. We will need to handle this through by either oversampling or SMOTE'ing to bring the minority class in proportion with the majority class.

Given the small proportion of the 'function needs repair' class we will combine the 'functional needs repair' and 'non functional' class together to create the new class 'needs attention'. Then we will cast the 'functional' class of wells as 'does not need attention'. 

# Data Cleaning

## Amount TSH: Illogical Values

Most total static head measurements are in the range of hundreds to tens of thousands.

Depths exceeding 50,000 units (e.g., feet or meters) are rare or unusual so we will filter these out as they are outliers. 

Most residential wells range from a few meters to around 100 meters (10 to 300 feet).

Wells used for industrial or agricultural purposes may have deeper total static heads, often ranging from tens to several hundred meters (hundreds to thousands of feet). 

## Longitude and Latitude: Zero Value Coordinates

Since these wells have longitude and latiude of 0 it is safe to say that these wells while they were recorded either have an unknown location or do not have a location inside of Tanzania and as such are not reliable data point.

We will be dropping all of these rows from the dataset.

## Installer Column: Typos

## Formatting Cleanup

## Missing Values

### Categorical columns with 0's

## Applying the correct data types

The columns listed below act more as lables than as integers. They identify a specific aspect of about the row entry and as such should be considered categorical variables.

## Missing values

### Filling Missing Values

Since all of the columns above are categorical columns, we will be filling them with the string 'unknown' so that we can maintain as much data as possible. 

## Dropping Columns

We are going to drop the following columns:

- 'recorded_by'
    - All records were recorded by the same person. No unique data is provided.
- 'quantity_group'
    - This column is identical to 'quantity'
- 'date_recorded'
    - The date that the information was recorded does not hold valuable information for determining the status of the well. 
- 'num_private'
    - There is no description for this column inside of the data table's documentation. 
    - https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/

# Baseline Feature Importance

## Baseline Feature Importance: Decision Tree

## Baseline Feature Importance: RFE

## Baseline Feature Importance: Analysis

Now that we have combined the feature selections together we can filter the dataframe so we can see the featuresthat are less likely to contribute to our models performance.

We will want to look at any feature that matches both of the criteria listed below and consider dropping those features. 

For RFE (Recursive Feature Elimination) features with higher ranks are considered less important by RFE so we will look at features with a rank below 20. 

For the decision tree feature importance, we will look at features that have a gini impurity below .01. 

### 'source_class' and 'source_type'

Even though this a low scoring column, the water source of the well is most likely a good indicator of the status of the well itself. We will want to retain as much information as we can that can lead us to determining the functional status of a well.

### 'payment_type'

This column is essentially the same as 'payment' as such we can drop this column.

### 'public_meeting'

Given its low scores and that there is no further information given in the documentation aside from True/False, we will drop this column.

### 'management_group'

- How the waterpoint is managed?

This could indicate how different management practices that affect the functionality of the well. As such, we will want to keep this column.

### 'water_quality'

While there is less crossover inside of water_quality and quality_group we will want to keep both. 

The water quality of the well is most likely one of the best indicators of the status of the well. 

We will want to retain as much information as we can that can lead us to determining the functional status of a well.

### Dropping columns

## Columns with too many unique values

Given the high amount of unique values that exist inside of various columns, we will bucket the values that fall under 1% of the columns total values. 

We will only be bucketing values for columns that have over 30 unique values so that we  maintain all of the unique values in columns that have smaller amount of unique values. 

# Column Correlation

## Numeric Columns

While some columns border on being too correlated with each other (~0.80), none of columns exceed this threshold. If a column had multiple high correlations (either positive or negative) with other columns we would drop that column, but this is not the case. 

# Scoring Metric

Inside of all the models listed below, we are using recall as our scoring metric. 

In the target variable, we started with three classes:
- Functional
- Non-Functional
- Functional needs repair

We combined the non-functional and the functional needs repair classes due to the heavy class imbalance inside of the functional needs repair class. Since functional needs repair and non-functional wells could both be classified into a needs attention class we combined them. Our updated classes are:

- Does not need attention (for functional wells)
- Needs attention (for non-functional and functionally needs repair wells)

As such, here is a breakdown of our confusion matrix:
- True Positive (TP): Correctly predicting a well as "Does not need attention".
- False Positive (FP): Incorrectly predicting a well as "Needs attention" when it actually does not need attention.
- True Negative (TN): Correctly predicting a well as "Needs attention".
- False Negative (FN): Incorrectly predicting a well as "Does not need attention" when it actually needs attention.

We chose to use recall as our primary scoring metric because it measures the proportion of wells that need attention that are correctly predicted as being in need of attention.

# Base Logistic Regression Models

# Base Decision Tree Models

# Lasso Regression: Feature Selection

## Feature Selection

Lasso Regression set the coefficient of 62 features to 0 indicating that these features are not considered important for predicting the target variable.

## Lasso Features: Logistic Regression

## Lasso Features: Decision Tree

# VIF: Feature Selection

## Feature Selection

## VIF Features: Logistic Regression

## VIF Features: Decision Tree

# Current Model Results

So far, the best model is the base model with undersampling to account for the class imbalance. 

We were hoping that either the VIF or Lasso feature selection would result in a better model but it has not. 

It is worth noting that the base decision tree given that there is no max depth set the model is most likley overfitting. 

However, we would like to perform dimensionality reduction on like-columns to determine if we can get better scores. 

# Dimensionality Reduction

Various columns contain similar information. While these columns are not a equal in every regard, they are similar enough as to where our models may perform better if we removed these features. We had originally kept these features as we thought having the extra data points inside of these features would be helpful for classification. 

If the columns contain over 80% of shared row values, we will drop a column. 

We will be comparing the following features:

- 'scheme_management' and 'management'
- 'region' and 'region_code'
- 'extraction_type' and 'extraction_type_group'
- 'water_quality' and 'quality_group'
- 'source' and 'source_type'
- 'waterpoint_type' and 'waterpoint_type_group'

Given the results above, we will take a closer look at the following pairs:

- 'scheme_management' and 'management'
- 'extraction_type' and 'extraction_type_group'
- 'waterpoint_type' and 'waterpoint_type_group'

We will also take a look at 'region' and 'region_code' seperately as while the values are not the same, they contain similar information. 

## 'scheme_management' and 'management'

Since management has fewer unknown values, we will drop the 'scheme_management' column from the DataFrame. 

## 'extraction_type' and 'extraction_type_group'

Since the 'extration_type' column is more specific than 'extraction_type_group', we will be dropping 'extraction_type_group' from the DataFrame. 

## 'waterpoint_type' and 'waterpoint_type_group'

Since the 'waterpoint_type' column is more specific than 'waterpoint_type_group', we will be dropping 'waterpoint_type_group' from the DataFrame. 

## 'region' and 'region_code'

Since the 'region_code' contains more values than the region, we will be dropping 'region' from the DataFrame.

## Dropping columns

# Dimensionality Reduction: Logistic Model

# Dimensionality Reduction: Decision Tree 

# Review Model Results

Since we have two different models, we will hyper-parameter tune the best two types of models to determine which is the best classifier. 

We will hyper-parameter tune the following models to determine which model performs best on the validation fold. 
- DecisionTree - VIF - Oversampling
- LogisiticRegression - base - Oversampling

Once determined, we will run the best model on the test data. 
