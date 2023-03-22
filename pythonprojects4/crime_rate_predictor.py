import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

# The task is to build a decision tree model to predict crime rate accuarately
# It is important to note that this is a regression problem as it predicts a continous outcome
# The first step is to load the dataframe to see what we're dealing with
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/'
                 'Standard_Metropolitan_Areas_Data-data.csv')
print(df)

# We then proceed to see if there are missing values in the dataframe
print(df.isnull().sum())
# Fortunately there are no missing values in our dataframe so no cleaning should be done

# We then seperate the numeric and categorical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
print(numerical_variables)
# All variables appear to be numerical hence no need to perform one-hot encoding to convert categorical variables to
# numeric

# We then seperate the important variables from the rest
# These are variables that can be used to accurately predict crime rate which is our target variable
df1 = df[['graduates', 'work_force', 'income', 'percent_city', 'crime_rate', 'region']]

# We then proceed to seperate the target variables from the input(predictor variables)
print('The input variables are')
x = df1.drop('crime_rate', axis=1)
print(x)
print('The target variable is')
y = df1['crime_rate']
print(y)

# We then proceed to split the input and target variables data into test and train datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train)
print(x_test)

# From this test and train data we can now start building our decision tree
# Since this is a regression problem we use the regression type decision tree
model = DecisionTreeRegressor(random_state=1)
data = model.fit(x_train, y_train)
print(data)

# We now use the model to predict crime rate
print('The first five are')
print(x.head())
print('The predictions are')
print(model.predict(x.head()))

# We want to see how the model looks like here we will use a module known as graphviz
dot_data = export_graphviz(model, out_file=None, feature_names=x_test.columns, class_names=['0', '1'],
                           filled=True, rounded=True, special_characters=True)


graph1 = graphviz.Source(dot_data, format='png')
graph1.render(view=True)

# After building the model we need to check its perfomance, this is measured using the cost function
# This is to see if the model works well or not
# This is by quantifying the difference between predicted and expected values to a real number
# Predicted value is the value predicted value of the machine learning model
# Expected value is the actual value of the data
# Cost functions can either be maximized or minimized
# Maximized cost functions usually returns values as large as possible
# Minimized cost functions usually aim to return values as small as possible
