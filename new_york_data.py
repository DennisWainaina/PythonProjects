# Loading important libraries
import pandas as pd
import numpy as np

# Loading the dataframe
df = pd.read_csv("AB_NYC_2019.csv")
print(df)

# Seeing the columns in the dataframe
print(df.columns)

# Seperating the numerical and categorical variables
# First the numerical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
print('These are the numerical variables')
print(numerical_variables)
# Then the categorical variables
categorical_variables = list(set(df.columns) - set(numerical_variables))
print('These are the categorical variables')
print(categorical_variables)

# Cleaning the numerical and categorical variables
# First the numerical variables
print(df[numerical_variables].isnull().sum())  # Reviews per month column has 10052 missing values
print(df['reviews_per_month'])
# Filling missing numerical values with mean
df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].mean())
print(df[numerical_variables].isnull().sum())  # Cleaning worked
# Then the categorical variables
print(df[categorical_variables].isnull().sum())  # Categorical variables has missing values
# Before cleaning
print('Before cleaning')
print(df[categorical_variables])
# Filling missing categorical variables with mode
df[categorical_variables] = df[categorical_variables].fillna(df[categorical_variables].mode().iloc[0])
print(df[categorical_variables].isnull().sum())  # Cleaning worked
print('After cleaning')
print(df[categorical_variables])
# Cleaning worked for both numerical and categorical variables
