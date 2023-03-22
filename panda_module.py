# This is a module for dealing with dataframes
# This is data containing rows and columns like an Excel file or csv file
# Random pieces of data is gotten from various info and analysed to give various information using this library
# It is the primary module used in Python for data analysis and data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# These are the columns for the dataframe which are immutable
g7_pop = pd.Series(
    [2.6, 2.7, 2.8, 2.9, 3.6, 3.1, 3.2],
    index=['Canada', 'France', 'Germany', 'Italy', 'Japan', 'UK', 'US']

                   )
g7_pop.name = 'Population in millions'
print(g7_pop)
print(g7_pop.values)
print(g7_pop.dtype)

certificates_earned = pd.Series(
    [8, 2, 5, 6],
    index=['Tom', 'Kris', 'Ahmad', 'Beau']
)

print(certificates_earned)

print(g7_pop['Canada'])

# It is important to note that in pandas library the upper limit is included for example
print(g7_pop['Canada':'Italy'])

# CSV files are Excel tables basically
print(g7_pop.describe())
# This command is used to give a general summary of the statistics of your data.

# To get the position of a specific element we use the iloc command for example;
print(certificates_earned.iloc[2])
# loc and iloc are used to select rows and just stating is used to select the columns

# We can also use boolean conditions like in numpy arrays to get data which matches a specific condition
# This is by using the loc command with the boolean condition in them example shown below


# Note there's a difference between a series and a dataframe here's an example of a dataframe
certificates_earned = pd.DataFrame({
    'Certificates': [8, 2, 5, 6],  # Certificates is the first column of the dataframe
    'Time (in months)': [16, 5, 9, 12]  # Time in months is the second column of the database
})

certificates_earned.index = ['Tom', 'Kris', 'Ahmad', 'Beau']  # These are the rows of the dataframe using index
print(certificates_earned)

# We can use boolean conditions to filter out unnecessary info for example:
print(certificates_earned.loc[certificates_earned['Certificates'] > 2])

# We can also filter out for specific columns as shown below
print(certificates_earned.loc[certificates_earned['Certificates'] > 2, 'Certificates'])

# It is important to note that panda operations are immutable such that they can't be changed

# We can also drop info while printing instead of selecting info using boolean operations for example:
print(certificates_earned.drop('Tom'))
# This prints the results of all the students except Tom

# One can also add columns with values to existing data for example:
longest_streak = pd.Series([13, 11, 9, 7], index=certificates_earned.index)  # Index is for the rows for the column
certificates_earned['Longest streak'] = longest_streak
print(certificates_earned)

# Similar to Excel we can create a new column from operations done on two existing columns for example:
certificates_earned['Certificate per month'] = certificates_earned['Time (in months)'] /\
                                               certificates_earned['Certificates']

print(certificates_earned)
print(certificates_earned.mean())  # Used to find the mean of each column in the data

plt.plot(certificates_earned)
plt.show()
# These two lines of code above were to check how the data appeared when plotted

# We can also do what is called cleaning the data which is checking for missing values from the data
# We also need to check the values such that they are relevant for the domain we are working with
# isnull or isnan checks whether a value is null or there is no value for example:
print(pd.notnull(0))

# We can also check all the null values in a series or dataframe using the sum function of null or notnull
# For example
print(pd.notnull(certificates_earned).sum())
# As can be seen in the series above there are 4 values for certificates for example
print(pd.notnull(longest_streak).sum())

s = pd.Series(['a', 3, np.nan, 1, np.nan])

print(s.notnull().sum())
print(longest_streak.notnull().sum())
# The code above is another way of writing the code in line 88

# One can also clean the data for an entire dataframe step by step for example:
print(certificates_earned.info())
# The command above is used to check on data on the dataframe for the type and if there are null values on columns

# We can also test this command on a dataframe with actual null values to see if it works for example:
df = pd.DataFrame({
    'Column A': [1, np.nan, 30, np.nan],
    'Column B': [2, 8, 31, np.nan],
    'Column C': [np.nan, 9, 32, 100],
    'Column D': [5, 8, 34, 110]
})
df.index = ['First', 'Second', 'Third', 'Fourth']
# First we print the dataframe to see if everything is correct
print(df)

# We then check for null values using the info method for example:
print(df.info())

# We can also check the total no of null values in each column of the dataframe using the is null method
print(df.isnull().sum())

# After identifying the null values the next step is to clean the data by fixing the null values
# To do this we use the four fill method and the backwards fill method for example:
print(df.fillna(method='ffill', axis=0))
# The axis 0 is used to show that the forward fill is column based (vertical based)

# We can also do it row based (horizontal based) by setting the axis equals to 1 as shown below:

year = df.fillna(method='ffill', axis=1)
print(year)
print(year.fillna(method='bfill', axis=1))

# One can also find duplicated data in a dataframe using the duplicate method
# By default python chooses the first option as duplicate but one can change the parameter such that it chooses last
# For example:
ambasaddors = pd.DataFrame({
    'Country': ['France', 'United Kingdom', 'United Kingdom', 'Italy', 'Germany', 'Germany', 'Germany']
})
ambasaddors.index = ['Gerard A.', 'Kim D.', 'Peter W.', 'Armando V.', 'Peter W.', 'Peter A.', 'Klaus S.']
print(ambasaddors)
print(ambasaddors.duplicated())

# One can then drop the values which have been duplicated by using the drop method for example:
print(ambasaddors.drop_duplicates())

# It is possible to visualize the data on a series or a dataframe
x = np.arange(-10, 11)
# We are now done with this tab on to reading csv files and text files
# You can also fill null values in a dataframe with a specific value using the fillna method
