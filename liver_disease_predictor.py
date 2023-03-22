# In this project we're going to be building a logistic regression model to determine if a patient has liver disease
# First we load important libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# First we load the data from an external source
liver_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/'
                         'liver_patient_data/indian_liver_patient_dataset.csv')
print(liver_data)

# Check if there are missing values in the dataset
print(liver_data.isnull().sum())

# Cleaning data for the Albumin_and_Globulin_Ratio coliumn
# First we check if its numerical or categoircal to see how to fill the missing values
print(liver_data['Albumin_and_Globulin_Ratio'])  # Values are numerical
# Filling in the missing values
liver_data['Albumin_and_Globulin_Ratio'] = liver_data['Albumin_and_Globulin_Ratio'].fillna\
    (liver_data['Albumin_and_Globulin_Ratio'].mean())
# Checking to see if it worked
print(liver_data.isnull().sum())  # No more null values

# Splitting the data into numerical and categorical variables
# First the numerical variables
numerical_variables = liver_data.select_dtypes(include=np.number).columns.tolist()
print(numerical_variables)
# Then the categorical variables
categorical_variables = list(set(liver_data.columns) - set(numerical_variables))
print(categorical_variables)

# Performing one-hot encoding on the categorical variables
df_dummy = pd.get_dummies(liver_data, prefix_sep='_', drop_first=True)
print(df_dummy)
print(df_dummy.columns)
liver_data = df_dummy
print(liver_data.columns)

# Splitting the data into target and input variables
x = liver_data.drop('Liver_Problem', axis=1)
y = liver_data['Liver_Problem']
print(x)  # Input variables
print(y)  # Target variables

# Splitting the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(x_train)
print(x_test)

# Building the model
model = LogisticRegression()
model.fit(x_train, y_train)
print(model)

# Using our model to predict data
y_predict = model.predict(x_test)
print(y_predict)

# Seeing accuracy of our model
accuracy = accuracy_score(y_test, y_predict)
print('The accuracy of the model is', accuracy)  # 0.648
MSE = mean_squared_error(y_test, y_predict)
print('The mean squared error is', MSE)  # 0.352
MAE = mean_absolute_error(y_test, y_predict)
print('The mean absolute error is', MAE)  # 0.352
R2 = r2_score(y_test, y_predict)
print('The R2 score is', R2)  # -0.542
recall = recall_score(y_test, y_predict)
print('The recall score is', recall)

# Let's see what happens when we use random forest ensemble method
# This creates many decision trees and makes a decision based on the answer given by the many decision trees
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(x_train, y_train)
new_predictions = rfc.predict(x_test)
print(new_predictions)

# Comparing the two accuracies to see if there has been an improvement
accuracy = accuracy_score(y_test, new_predictions)
print('The accuracy of the ensemble model is', accuracy)  # 0.691
MSE = mean_squared_error(y_test, new_predictions)
print('The mean squared error of the ensemble model is', MSE)  # 0.309
MAE = mean_absolute_error(y_test, new_predictions)
print('The mean absolute error of the ensemble model is', MAE)  # 0.309
R2 = r2_score(y_test, new_predictions)
print('The R2 score of the ensemble model is', R2)  # -0.356
recall = recall_score(y_test, new_predictions)
print('The recall score of the ensemble model is', recall)
precision = precision_score(y_test, new_predictions)
print('The precision score is', precision)
# The accuracy seems to have improved due to the using of ensemble methods

# We can try to deal with class imbalance where there is a lot of data of one type compared to another
sm = SMOTE(random_state=25, sampling_strategy=1.0)  # again we are eqalizing both the classes
# Fitting the data into the new model which has solved class imbalance
x_train, y_train = sm.fit_resample(x_train, y_train)

# Now building model based on this new data which has been corrected for class imbalance
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
newer_predictions = rfc.predict(x_test)
print(newer_predictions)

# Checking accuracy of model
accuracy = accuracy_score(y_test, newer_predictions)
print('The accuracy of the ensemble model after SMOTE is', accuracy)
MSE = mean_squared_error(y_test, newer_predictions)
print('The mean squared error of the ensemble model after SMOTE is', MSE)
MAE = mean_absolute_error(y_test, newer_predictions)
print('The mean absolute error of the ensemble model after SMOTE is', MAE)
R2 = r2_score(y_test, newer_predictions)
print('The R2 score of the ensemble model after SMOTE is', R2)

# Building decision tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

# Testing the model
accuracy = accuracy_score(y_test, y_prediction)
print('The accuracy of the logistic regression model after SMOTE is', accuracy)
MSE = mean_squared_error(y_test, y_prediction)
print('The mean squared error of the logistic regression model after SMOTE is', MSE)
MAE = mean_absolute_error(y_test, y_prediction)
print('The mean absolute error of the logistic regression model after SMOTE is', MAE)
R2 = r2_score(y_test, y_prediction)
print('The R2 score of the logistic regression model after SMOTE is', R2)
f1 = f1_score(y_test, y_prediction)
print('The f1 score is', f1)
roc = roc_auc_score(y_test, y_prediction)
print('The roc score is', roc)

# Finding the true negative rate
cm = confusion_matrix(y_test, newer_predictions)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TNR = TN/(FP+TN)
print('The True Negative rate is', TNR)