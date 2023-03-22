import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/fraud_data.csv')
print(df)
print(df['id_34'].isnull().sum())
print(df.columns)


def cleaning():
    # First step is pre-processing which is the preparing of data for training.
    # Data may have missing values, wrongful data entries, class imbalance and diff. scales of data
    # This may include: Handling missing values, standardization and handling imbalanced data sets
    # One of the methods used to fill in missing data is the use of single imputation
    # This is the filling of missing values with the mode or mean of the data
    # Mean if the variable is numeric and mode if the variable is categorical for example:
    print(df['DeviceType'].mode())
    # Since the value is categorical we find the mode of the data first
    # We then fill the missing value with the mode of the data

    df['DeviceType'] = df['DeviceType'].fillna(df['DeviceType'].mode()[0])
    # With the code above the missing values have been filled with the most common value which is desktop
    print(df)
    # We know  do the same for the other variables with missing values
    print(df['DeviceInfo'].mode())
    df['DeviceInfo'] = df['DeviceInfo'].fillna(df['DeviceInfo'].mode()[0])
    print(df)

    # Repeating the same process for the id_38 variable
    print(df['id_38'].mode())
    df['id_38'] = df['id_38'].fillna(df['id_38'].mode()[0])
    print(df)

    # For the id_37 variable
    print(df['id_37'].mode())
    df['id_37'] = df['id_37'].fillna(df['id_37'].mode()[0])
    print(df)

    # For the id_36 variable
    print(df['id_36'].mode())
    df['id_36'] = df['id_36'].fillna(df['id_36'].mode()[0])
    print(df)

    # For the id_35 variable
    print(df['id_35'].mode())
    df['id_35'] = df['id_35'].fillna(df['id_35'].mode()[0])
    print(df)

    # For the id_34 variable
    print(df['id_34'].mode())
    df['id_34'] = df['id_34'].fillna(df['id_34'].mode()[0])
    print(df)

    # For the id_33 variable
    print(df['id_33'].mode())
    df['id_33'] = df['id_33'].fillna(df['id_33'].mode()[0])
    print(df)

    # For the id_32 variable
    print(df['id_32'].mode())
    df['id_32'] = df['id_32'].fillna(df['id_32'].mode()[0])
    print(df)

    # For the id_31 variable
    print(df['id_31'].mode())
    df['id_31'] = df['id_31'].fillna(df['id_31'].mode()[0])
    print(df)


cleaning()
# It is important to understand that imputation does not come without consequences
# One is advised to only use imputation if less than 20% of all values are null
# We now check for missing variables in the columns for example:
print(df['DeviceType'].isnull().sum())
print(df['DeviceInfo'].isnull().sum())
# The output is 0 so it worked

print('The fraud values are;')
print(df['isFraud'][df['isFraud'] == 1])

# Now checking for numerical variables
print(df['TransactionDT'].isnull().sum())
print(df['card5'].isnull().sum())
print(df['card5'])

# Cleaning the numerical variable
plt.hist(df['card5'])
plt.title('Histogram before cleaning')
plt.show()
df['card5'] = df['card5'].fillna(df['card5'].mean())
print('The number of null values is', df['card5'].isnull().sum())
print(df['card5'])
plt.hist(df['card5'])
plt.title('Histogram after cleaning')
plt.show()

# There is a way to know all the numeric and categorical variables for example:
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
# These are now the numerical variables
print(df['id_32'])
print(numerical_variables)

# With the knowledge of the numeric variables with numeric variables we can fill them with their means in one line
df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].mean())
print(df[numerical_variables])
print(df[numerical_variables].isnull().sum())

# We can also do the same for the categorical variables instead of going through them one by one like above
categorical_variables = list(set(df.columns) - set(numerical_variables))
print(categorical_variables)
df[categorical_variables] = df[categorical_variables].fillna(df[categorical_variables].mode().iloc[0])
print(df[categorical_variables].isnull().sum())
print(df['card5'].describe())

# There is something known as one-hot encoding in Python
# This occurs because machine learning model expects inputs to be numbers
# It is therfore a must to convert categorical variables to numeric variables
# This is known as one-hot encoding
# The way it works is it assigns 1 to the target categorical variable if yes in question and 0 to the other variables
# The machine learning algorithm cannot work with categorical variables, but we need to use them for prediction.
# We must therefore convert them to numerical variables first before proceeding with machine learning
# The process is as follows;
print(df.info()) 

# We first isolate the categorical variable and get samples which we perform one-hot encoding
# This is shown in the code below which introduces new variables and 1 if it is and 0 if its not
df_dummy = pd.get_dummies(df, prefix_sep='_', drop_first=True)
print(df_dummy)
print(df_dummy.info())
# As can be seen the no of variables increased form 434 to 1636 before machine learning process takes place.
# This is because it splits the categorical variables to 1 for yes and 0 for no before feeding to ML algorithm.
print(df)
# As can be seen above the original dataframe is unaffected as we didn't assign it to the new variable

# An example of a machine learning model is the decision tree which will be explained briefly
# Without going into too much detail the idea is it seperates the variables with true or false(booleans)

# Machine learning is classified into two types : classification and regression
# Classification is process of classifying outcome into two or more types
# An example is the determination of whether a transaction is fraud or not or whether color is r, b or g
# Regression is the process of predicting a continous variable at a specific point in time
# An example is the prediction of the price of a home 
# Another example is the prediction of the age of a dog based on dog images

# It is possible to have class imbalance in a machine learning algorithm.
# This is a classification problem where there is too much data of one type compared to another.
# An example is too much of men compared to women as the data being fed to the ML algorithm.
# This leads to inaccurate predictions as the machine has too much data of one type.
# The task thus becomes to balance the data for more accurate predictions
# This is done by either oversampling, undersampling or SMOTE.
# Oversampling is increasing minority data undersampling is decreasing majority data.
# SMOTE is a combination of both to obtain a certain ratio which can be used in the ML algorithm.
