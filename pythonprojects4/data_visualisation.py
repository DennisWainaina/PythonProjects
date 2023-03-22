import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Standard_Metropolitan_Areas_Data-data.csv')
print(df.head())

# Scatter plots use dots to represent value of various numeric data for example:
plt.scatter(df['region'], df['crime_rate'])  # The first part is the x-axis and the second y-axis
plt.xlabel('region')
plt.ylabel('Crime rate as percentage')
plt.title('Crime rates as a percentage in different regions')
plt.show()

# Scatter plots are used to spot patterns in data
# They can also be used to spot outliers in the data

# Line charts are used to represent data over a continous time span
# Shows trend of a variable over time for example:
plt.plot(df['work_force'], df['income'])
plt.xlabel('No of work force')
plt.ylabel('Income')
plt.title('How income relates to workforce')
plt.show()

# A histogram is a graphical display of data using bars of different heights
# The x-axis indicates the intervals which measurements fall under the and the y-axis is the frequency of occurence

# One can also use a bar graph to represent categorical data
# This is data split into categories in the x-axis and y-axis
# For example:
plt.bar(df['region'], df['crime_rate'])
plt.xlabel('region')
plt.ylabel('Crime rate as percentage')
plt.title('Crime rates as a percentage in different regions')
plt.show()

# We will now work with data from a dataset to see how to plot consecutive plots on Matplotlib for example:
plt.plot(df['work_force'], df['income'])
plt.show()
# This is the first consecutive plot

# Now on to the next consecutive plot
plt.plot(df['physicians'], df['income'])
plt.show()

# We can also combine the two plots if we use plt.show() at the end for example:
plt.plot(df['work_force'], df['income'], color='r', label='work_force')
plt.plot(df['physicians'], df['income'], label='physicians')
plt.xlabel('Income')
plt.ylabel('Workforce and Physicians')
# One can also add legends by defining a label as shown above and calling the legend function as shown below
plt.legend()
plt.show()

# One can change the size of the plots by using the plt.figure() method for example:
plt.figure(figsize=(12, 5))  # 12 by 5 plot
plt.plot(df['work_force'], df['income'])
plt.xlabel('No of work force')
plt.ylabel('Income')
plt.title('How income relates to workforce')
plt.show()

# One can have multiple plots in one figure using the subplot command for example:
plt.subplot(1, 2, 1)  # The numbers mean the row, column and index of the plot(order of printing) in that order
plt.plot(df['work_force'], df['income'])
plt.title('Income vs Work Force')
# This is the first plot

plt.subplot(1, 2, 2)  # The last two means it will be shown second
plt.plot(df['hospital_beds'], df['income'])
plt.title('Income vs Hospital beds')

# One can also add a centered title to the two plots using the plt.suptitle() command for example:
plt.suptitle('Sub Plots')
plt.show()

# Barplot af people according to regions
print(df['region'])
