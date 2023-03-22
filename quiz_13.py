# In this project we are building a linear regression model
# Loading important libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier

# Loading dataframe
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/kc_house_data/kc_house_data.csv')
print(df)

# Check if there are missing values
print(df.isnull().sum())  # No missing values

# Splitting columns into numerical and categorical varialbes
# First the numerical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
print(numerical_variables)
# Then the categorical variables
categorical_variables = list(set(df.columns) - set(numerical_variables))
print(categorical_variables)  # No categorical variables

# Splitting the data into input and target variables
x = df.drop('price', axis=1)
y = df['price']
print(x)
print(y)

# Splitting data into test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Building linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
print(model)

# Using model to predict test data
y_predict = model.predict(x_test)

# Seeing accuracy of model
r2 = r2_score(y_test, y_predict)
print('The r2 score is', r2)

# Building decision tree regressor model
model1 = DecisionTreeRegressor(max_depth=3)
model1.fit(x_train, y_train)
print(model)

# Using model to predict test data
y_predict = model1.predict(x_test)

# Seeing accuracy of model using r2 score
r2 = r2_score(y_test, y_predict)
print('The r2 score is', r2)

# Seeing RMSE of Decision Tree Regressor model with max depth of 3 and random state of 42
MSE = mean_squared_error(y_test, y_predict)
RMSE = np.sqrt(MSE)
print('The RMSE of the model with max depth of 3 and random state 42 is', RMSE)

# Seeing R2 score of the model above
r2 = r2_score(y_test, y_predict)
print('The R2 score of the model with max depth of 3 and random state 42 is', r2)

# Building random forest ensemble model
rfc = RandomForestClassifier(max_depth=6, n_estimators=10, random_state=42)
rfc.fit(x_train, y_train)
new_predictions = rfc.predict(x_test)
print(new_predictions)

# Checking RMSE of model
MSE = mean_squared_error(y_test, new_predictions)
RMSE = np.sqrt(MSE)
print('The RMSE of the random forest ensemble model with max depth of 3 and random state 42 is', RMSE)

# Checking MAE of model
MAE = mean_absolute_error(y_test, new_predictions)
print('The MAE of the random forest ensemble model with max depth of 3 and random state 42 is', MAE)

# Checking r2 score of the random forest ensemble model
r2 = r2_score(y_test, new_predictions)
print('The R2 score of the model with max depth of 3 and random state 42 is', r2)

# Building logistic regression model and seeing accuracy of model
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# Seeinag accuracy and recall score
accuracy = accuracy_score(y_test, y_predict)
print('The accuracy score is', accuracy)
recall = recall_score(y_test, y_predict, average='binary')
print('The recall score is', recall)

# Checking precision score
precision = precision_score(y_test, y_predict, average='binary')
print('The precision score is', precision)
