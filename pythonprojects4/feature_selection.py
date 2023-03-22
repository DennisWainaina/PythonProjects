# Feature selection algorithms are used to select important features that are used to be used in our model.
# This is to remove features that are irrelevant for prediction using our model.
# It uses various statistical models and calculations to determine important features that go into our model
# This is usually done after the test and train data are selected such that pre-processing(removing of irrelevant
# features) is done before the data is fed into the model.
# There are various feature selection algorithms such as filter, wrapping and intrinsic methods

# First we load important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy

# Then we load the database
col = ["num_preg", "glucose", "D_blood_pressure", "skin_fold_thickness", "serum_insulin",
       "body_mass_index", "pedigree_func", "age", "diabetes"]
diabetes_df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/ML_Models/master/Performance_Evaluation/"
                          "diabetes.txt", names=col)
print(diabetes_df)

# Then we create a correlation matrix to see how our variables relate with one another which can be positive, negative
# or even 0.
corr = diabetes_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="RdYlGn")
# plt.show()
# Using this heatmap one might think that one can just choose features based on intuition but that brings bias.

# Now selecting our input and target variables
x = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes']
print(x)  # Input variables
print(y)  # Target variables

# Splitting data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
print(x_train)
print(x_test)

# We now want to use feature selection which has various steps which include:
# 1. Choosing a base model with all features before selection.
# 2. Use feature selection algorithm to select the most important features.
# 3. Use these important features to create a new dataset using those features.
# 4. Create a new model based on these features.
# 5. Compare perfomance of the new model compared to the base model

# One of the feature selection techniques is recurssive feature elimination
# This feature selection algorithm runs again and again removing unimportant features at each stage
# For example using our dataset:

# 1. Building a base model before feature selection which in our case is a logistic regression model
base_model = LogisticRegression(random_state=1, solver='lbfgs', max_iter=400)
base_model.fit(x_train, y_train)
print(base_model)
# Using our model to predict data
y_predict = base_model.predict(x_test)
print(y_predict)
# Checking accuracy and f1 score of the base model
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print('The accuracy of the base model is ', accuracy)  # Accuracy is 0.77559
print('The f1 score of the base model is ', f1)  # f1 score is 0.6369
print('The r2 score of the base model is', r2)

# 2. We then use feature selection techniques to select important features
rfe = RFE(estimator=base_model, step=1)  # step=1 means we're removing one feature at a time
rfe.fit(x_train, y_train)
print(rfe)
# If we want to see the ranking of features we can do the following:
selected_rfe_features = pd.DataFrame({
    'Feature': list(x_train.columns),
    'Ranking': rfe.ranking_
})
# We can then sort the features in terms of ranking from 1 to 5
book = selected_rfe_features.sort_values(by='Ranking')
print(book)

# 3. We then use a dataset using these important features
# This is by using the transform method on the dataset
x_train_rfe = rfe.transform(x_train)
x_test_rfe = rfe.transform(x_test)
# We then build our base model based on the corrected data
base_model.fit(x_train_rfe, y_train)
# We use it to predict data
rfe_predictions = base_model.predict(x_test_rfe)
print(rfe_predictions)
# Checking accuracy and f1 score of the predictions made by the transformed model
accuracy = accuracy_score(y_test, rfe_predictions)
f1 = f1_score(y_test, rfe_predictions)
r2 = r2_score(y_test, rfe_predictions)
print('Accuracy of rfe corrected model is', accuracy)
print('F1 score of rfe corrected model is', f1)
print('R2 score of rfe corrected model is', r2)
# Accuracy, f1 score and r2 score have improved

# Our next base model will be a random forest ensemble model which is better than the normal models as a base model
base_model1 = RandomForestClassifier(n_estimators=10000, random_state=1, n_jobs=-1)
base_model1.fit(x_train, y_train)
y_predict = base_model1.predict(x_test)
print(y_predict)
# Selecting important features
selected_features = SelectFromModel(estimator=base_model1)
# Training selector
selected_features.fit(x_train, y_train)
# Now with our new data we build a model based on this data
# First transforming our data to include only important features
x_important_train = selected_features.transform(x_train)
x_important_test = selected_features.transform(x_test)
# Training our model based on these features

base_model1.fit(x_important_train, y_train)
y_predict1 = base_model1.predict(x_important_test)
# We now want to see the accuracy of the random forest ensemble model with and without selecting important features
# First the accuracy of the random forest ensemble model which is the base model
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
print('Accuracy of random forest classifier without feature selection is', accuracy)
print('F1 score of random forest classifier without feature selection is', f1)
new_accuracy = accuracy_score(y_test, y_predict1)
new_f1 = f1_score(y_test, y_predict)
print('Accuracy of random forest classifier with feature selection is', new_accuracy)
print('F1 score of random forest classifier with feature selection is', new_f1)

# Another feature selection method we can use is called boruta method
# This captures all the features with info relevant for prediction
# First step is building the base model which is a random forest ensemble model
forest = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=1)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
# Checking the accuracy of the base model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Base model accuracy is', accuracy)
print('F1 score of base model is', f1)

# Applying boruta algorithm to our model
boruta_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
# Finding important features
boruta_selector.fit(np.array(x_train), np.array(y_train))
# Transforming data using boruta to include important features
x_important_train = boruta_selector.transform(np.array(x_train))
x_important_test = boruta_selector.transform(np.array(x_test))
# Training our model based on these important features
rf_important = RandomForestClassifier(n_estimators=10000, random_state=1, n_jobs=-1)
rf_important.fit(x_important_train, y_train)
# Predicting data based on our new model
new_predictions = rf_important.predict(x_important_test)
# Seeing accuracy after boruta methods
accuracy = accuracy_score(y_test, new_predictions)
f1 = f1_score(y_test, new_predictions)
r2 = r2_score(y_test, new_predictions)
print('Accuracy after Boruta is', accuracy)
print('F1 score after Boruta is', f1)
print('R2 score after Boruta is', r2)
