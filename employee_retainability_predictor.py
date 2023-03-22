import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix

# Load libraries and data.
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/HR_comma_sep.csv')
print(df)

# Do some exploratory data analysis to figure out which variables have a direct and clear impact on employee retention
# Impact of employee salaries on retention
# See the correlation between department and employee retention
# Separate dependent and independent variables.
# Split the data into train set and test set
# Now build a Logistic Regression model and make predictions for test data
# Measure the accuracy of the model

# First doing a bit of experimental data analysis
print(df.columns)
print(df['left'])
print(df['left'][df['left'] == 1])
print('The people who left the company are', len(df['left'][df['left'] == 1]))
print(df['salary'])
print(df['salary'][df['salary'] == 'high'])
print(df['salary'][df['salary'] == 'low'])
print(df['salary'][df['salary'] == 'medium'])
print(df['Work_accident'][df['Work_accident'] == 1])
print(df['satisfaction_level'][df['satisfaction_level'] <= 0.6])

# Checking if there are null values
print(df.isnull().sum())
# There are no null values hence no need for cleaning

# Splitting the data into categorical and numerical variables
print("These are the numeric variables")
numeric_variables = df.select_dtypes(include=np.number).columns.tolist()
print(numeric_variables)

# Now the categorical variables
categorical_variables = list(set(df.columns) - set(numeric_variables))
print('These are the categorical variables')
print(categorical_variables)


# Converting categorical variables to numeric using one-hot encoding
df_dummy = pd.get_dummies(df[categorical_variables], prefix_sep='_', drop_first=True)
print(df_dummy)
print(df_dummy['salary_medium'])
print(df_dummy.columns)

# Choosing target and input variables
df1 = pd.concat([df[numeric_variables], df_dummy], axis=1)
x = df1.drop('left', axis=1)
y = df1['left']
print(x)
print(y)

# Creating the training and test data from input and target variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
print(x_train)
print(x_test)

# Creating the model
Logisticmodel = LogisticRegression()
Logisticmodel.fit(x_train, y_train)
print(Logisticmodel)

# Making predictions for test data
y_predict = Logisticmodel.predict(x_test)
print(y_predict)
print(len(y_predict))
print('The people who stayed are', len(y_predict[y_predict == 0]))

# Testing accuracy of predictions hence how good the model built is
accuracy = accuracy_score(y_test, y_predict)  # Using accuracy score
print('The accuracy is', accuracy)
MSE = mean_squared_error(y_test, y_predict)
print('The mean squared error is', MSE)  # Using mean squared error
MAE = mean_absolute_error(y_test, y_predict)
print('The Mean Absolute Error is', MAE)  # Using mean absolute error
RMSE = np.sqrt(MSE)
print('The root mean squared error is', RMSE)  # Using Root mean squared error
r2 = r2_score(y_test, y_predict)
print('The r2 score is', r2)

# Calculating confusion matrix to see the data
cm = confusion_matrix(y_test, y_predict)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
print(conf_matrix)
print(df['time_spend_company'])

# Effect of employee salary on retention
high_paid_employees = df['salary'][df['salary'] == 'high']
print('High paid employees that left are', len(high_paid_employees[df['left'] == 1]))
medium_paid_employees = df['salary'][df['salary'] == 'medium']
print('Medium paid employees that left are', len(medium_paid_employees[df['left'] == 1]))
low_paid_employees = df['salary'][df['salary'] == 'low']
print('Low paid employees that left are', len(low_paid_employees[df['left'] == 1]))
print('The number of high paid employees are', len(high_paid_employees))
print('The number of medium paid employees are', len(medium_paid_employees))
print('The number of low paid employees are', len(low_paid_employees))
print('The number of employees who leave increase with decrease in salary ')
print('Therefore a decrease in salary causes a decrease in employee retention')
