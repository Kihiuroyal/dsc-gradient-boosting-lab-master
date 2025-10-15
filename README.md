# Gradient Boosting - Lab

## Introduction

In this lab, we'll learn how to use both **Adaboost** and **Gradient
Boosting** classifiers from scikit-learn!

------------------------------------------------------------------------

## Objectives

You will be able to: - Use **AdaBoost** to make predictions on a
dataset. - Use **Gradient Boosting** to make predictions on a dataset.

------------------------------------------------------------------------

## Getting Started

In this lab, we'll use boosting algorithms to classify data from the
**Pima Indians Diabetes Dataset** (`pima-indians-diabetes.csv`).\
Our goal is to use boosting algorithms to determine whether a person has
diabetes.

### Import Libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
```

------------------------------------------------------------------------

## Data Import and Exploration

``` python
# Import the data
df = pd.read_csv('pima-indians-diabetes.csv')

# Print the first five rows
df.head()
```

------------------------------------------------------------------------

## Cleaning, Exploration, and Preprocessing

The target variable is **'Outcome'**, where `1` denotes a patient with
diabetes.

### Steps:

1.  Check for missing values.
2.  Count patients with and without diabetes.
3.  Separate the target column and remove it from the dataset.
4.  Split the data into training and test sets (test_size=0.25,
    random_state=42).

``` python
# Check for missing values
df.isnull().sum()

# Number of patients with and without diabetes
df['Outcome'].value_counts()

target = df['Outcome']
df = df.drop('Outcome', axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)
```

------------------------------------------------------------------------

## Train the Models

``` python
# Instantiate models
adaboost_clf = AdaBoostClassifier(random_state=42)
gbt_clf = GradientBoostingClassifier(random_state=42)

# Fit models
adaboost_clf.fit(X_train, y_train)
gbt_clf.fit(X_train, y_train)
```

------------------------------------------------------------------------

## Make Predictions

``` python
# AdaBoost predictions
adaboost_train_preds = adaboost_clf.predict(X_train)
adaboost_test_preds = adaboost_clf.predict(X_test)

# GradientBoosting predictions
gbt_clf_train_preds = gbt_clf.predict(X_train)
gbt_clf_test_preds = gbt_clf.predict(X_test)
```

------------------------------------------------------------------------

## Model Evaluation

``` python
def display_acc_and_f1_score(true, preds, model_name):
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    print("Model: {}".format(model_name))
    print("Accuracy: {}".format(acc))
    print("F1-Score: {}".format(f1))
    
print("Training Metrics")
display_acc_and_f1_score(y_train, adaboost_train_preds, model_name='AdaBoost')
display_acc_and_f1_score(y_train, gbt_clf_train_preds, model_name='Gradient Boosted Trees')

print("Testing Metrics")
display_acc_and_f1_score(y_test, adaboost_test_preds, model_name='AdaBoost')
display_acc_and_f1_score(y_test, gbt_clf_test_preds, model_name='Gradient Boosted Trees')
```

### Confusion Matrices & Reports

``` python
adaboost_confusion_matrix = confusion_matrix(y_test, adaboost_test_preds)
gbt_confusion_matrix = confusion_matrix(y_test, gbt_clf_test_preds)

print(confusion_matrix(y_test, adaboost_test_preds))
print(confusion_matrix(y_test, gbt_clf_test_preds))

print(classification_report(y_test, adaboost_test_preds))
print(classification_report(y_test, gbt_clf_test_preds))
```

**Question:** How did the models perform? Interpret the metrics and
write your answer below.

------------------------------------------------------------------------

## Cross-Validation

``` python
print('Mean Adaboost Cross-Val Score (k=5):')
print(cross_val_score(adaboost_clf, df, target, cv=5).mean())
# Expected Output: 0.7631270690094218

print('Mean GBT Cross-Val Score (k=5):')
print(cross_val_score(gbt_clf, df, target, cv=5).mean())
# Expected Output: 0.7591715474068416
```

------------------------------------------------------------------------

## Summary

In this lab, we learned how to: - Use **AdaBoost** and **Gradient
Boosting** classifiers from scikit-learn. - Evaluate model performance
using accuracy, F1-score, confusion matrices, and cross-validation. -
Apply these ensemble methods on a real-world dataset for diabetes
prediction!
