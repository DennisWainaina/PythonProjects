import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn is a module which uses matplotlib to visualise data which uses less code to show the same figures as matplot

crash_df = sns.load_dataset('car_crashes')
print(crash_df)
# Plotting histograms using dataframes
year = sns.histplot(crash_df['not_distracted'], kde=False, bins=25)
plt.show()

# Plotting jointplots which compares two sets of data in a dataframe
print(sns.jointplot(x='speeding', y='alcohol', data=crash_df, kind='hex'))
plt.show()

# Plotting density plots of a set of data in a dataframe
print(sns.kdeplot(crash_df['alcohol']))
plt.show()

# Loading a dataset in the seaborn module
tips_df = sns.load_dataset('tips')
print(tips_df)

# Plotting pairplots of data sets in the dataframe
print(sns.pairplot(tips_df, hue='sex', palette='Blues'))
plt.show()

# Plotting rugplots which are denser where the value is most common
print(sns.rugplot(tips_df['tip']))
plt.show()

# One can also style graphs displayed for them to look differently for example
sns.set_style('darkgrid')

# These two pieces of code below are used to change the size and font of the graphs displayee below
plt.figure(figsize=(8, 4))
sns.set_context('talk', font_scale=1.4)

print(sns.jointplot(x='speeding', y='alcohol', data=crash_df, kind='reg'))
plt.show()

# One can also compare dataframes between two types of data using categorical plots such as barplots
# By default mean data is printed but one can use estimator to print for various functions for example
print(sns.barplot(x='sex', y='total_bill', data=tips_df, estimator=np.median))
plt.show()


# One can also count the number of occurences in a dataframe and display them using the countplot function
print(sns.countplot(x='sex', data=tips_df))
plt.show()

# There are also boxplots which compares two types of data sets on the y-axis with a type of data sets on the x-axis
# hue is the command used to compare the two different types of data
# The line in the middle is the median and the part extending above is one standard deviation from median
# The vertical line after the box is all other data while the dots are the outliers
print(sns.boxplot(x='day', y='total_bill', data=tips_df, hue='sex'))
plt.show()


# One can also use violin plots which is a variation of the box plot for example:
print(sns.violinplot(x='day', y='total_bill', data=tips_df, hue='sex', split=True))
plt.legend(loc=0)
# This code above is used to reposition the legend of the graph
plt.show()

# There are also stripplots which are used to seperate data and display them in terms of dots
# Dodge command is used to split men from women 
print(sns.stripplot(x='day', y='total_bill', data=tips_df, jitter=True, hue='sex', dodge=True))
plt.show()

# There are also swarmplots which are like violinplots
print(sns.swarmplot(x='day', y='total_bill', data=tips_df, color='green'))
plt.show()

# One can also use palettes to try and make graphs look better for example:
print(sns.swarmplot(x='day', y='total_bill', data=tips_df, hue='sex', palette='afmhot'))
plt.show()

# There are also special types of plots called matrix plots
# The first one is called heatmaps
plt.figure(figsize=(8, 6))
sns.set_context('paper', font_scale=1.4)
crash_mx = crash_df.corr()
print(sns.heatmap(crash_mx, annot=True, cmap='hot'))
plt.show()
