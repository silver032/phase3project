### Overview
This project uses data from Taarifa and the Tanzanian Ministry of Water to predict the functionality of water pumps throughout Tanzania. The model aims to categorize each pump as functional, functional but needs repair, or non-functional. This effort supports the initiative to ensure the availability of clean, potable water and facilitate effective maintenance operations.

# Dataset Description
The dataset includes several variables regarding the type of pump, its installation details, and how it is managed. These variables are crucial for predicting the operational status of the pumps, aiming to categorize them into three functional states.

# Business Problem 
The project is commissioned by UN-Water to identify factors that lead to well failures, thereby enhancing water availability and maintenance efficiency across Tanzania.

What impact will it have if we can predict well failures?
What are some of the leading features that lead to well failure or repair
What efficiencies can be improved across the Tanzania?

# Goals
Predictive Analysis: To predict the operational status of water pumps in three categories: functional, needs repair, and non-functional.
Indicator Analysis: To identify key indicators that lead to well malfunction or the need for repair.

# Key Questions
1. What preprocessing steps do we need to take to create an effective predictive model?
2. What encoding techniques are best fit for this?
3. Which type of predictive model gets the best results?
4. What hypertuning techniques are used to tune the model?


# Target Variable Investigation
In the target variable, we started with three classes:
Functional
Non-Functional
Functional needs repair

We combined the non-functional and the functional needs repair classes due to the heavy class imbalance inside of the functional needs repair class. Since functional needs repair and non-functional wells could both be classified into a needs attention class we combined them. Our updated classes are:

Does not need attention (for functional wells)
Needs attention (for non-functional and functionally needs repair wells)

As such, here is a breakdown of our confusion matrix:
True Positive (TP): Correctly predicting a well as "Does not need attention".
False Positive (FP): Incorrectly predicting a well as "Needs attention" when it actually does not need attention.
True Negative (TN): Correctly predicting a well as "Needs attention".
False Negative (FN): Incorrectly predicting a well as "Does not need attention" when it actually needs attention.

## Preprocessing Steps

# Data Cleaning: 
 Handling missing values and removing irrelevant features.

# Data Encoding: 
Applying appropriate techniques to handle categorical variables. Used OnehotEncoder

## Predictive Modeling

# Model Selection: 
Various models were evaluated including Decision Trees and Logistic Regression.

# Model Comparison:
Performance metrics (like Recall and Accuracy) were compared across different preprocessing strategies and models.

## Hyperparameter Tuning
Fine-tuned using grid search techniques to optimize performance, particularly focusing on improving recall to reduce the number of false negatives (functional pumps predicted as non-functional).

### Exploratory Data Analysis 

# Status of wells

# Cofusion Matrix

# Results of Modells 

image1
