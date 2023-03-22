import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/'
                 'Standard_Metropolitan_Areas_Data-data.csv')
print(df)
print(df.info())

# What is the mean area and maximum crime rate
print('The mean land area is', df['land_area'].mean())
print('The maximum crime rate is', df['crime_rate'].max())

# What is the minimum income in Millions of Dollars?
print('The minimum income in millions of dollars of all areas is', df['income'].min())

# The no of non-null entries in hospital beds and the region mostly occupied
print('The no of null entries in the hospital beds column is', df['hospital_beds'].notnull().sum())
print('The people who live in each region are', df['region'].value_counts())

# The no of people who live in region 4 and the average crime rates of all regions
fourth_region = df['region'].loc[df['region'] == 4]
print(fourth_region)
print('The people who live in region 4 are', fourth_region.notnull().sum())
print('The average crime rate is', df['crime_rate'].mean())

# The first five land areas of the metropolitan area data of region 4
df.to_excel('metropolitan_data.xlsx')
print('These are the first five land areas of region 4')
fourth_region_areas = df['land_area'][df['region'] == 4]
print(fourth_region_areas.head())

# Correlation between various values in the dataframe
matrix = df.corr()
print('The correlation matrix is shown below')
print(matrix)

# The no of areas in region 1 with crime rate greater than or equal to 54.16
first_region = df['region'].loc[df['region'] == 1]
lawless_areas = first_region.loc[df['crime_rate'] >= 54.16]
print('The no of areas in region 1 with crime greater than 54.16 is', lawless_areas.notnull().sum())

# The no of areas in region 3 with land area greater than or equal to 5000
third_region = df['region'].loc[df['region'] == 3]
big_area = third_region.loc[df['land_area'] >= 5000]
print('The answer is', big_area.notnull().sum())

# The no of Metropolitan areas in region 4 with crime rate is 85.62
fourth_region = df['region'].loc[df['region'] == 4]
high_crime_areas = fourth_region.loc[df['crime_rate'] == 85.62]
print('The answer is', high_crime_areas.notnull().sum())

# The no of Metropolitan areas in region 4 with crime rate is 55.64
middle_crime_areas = fourth_region.loc[df['crime_rate'] == 55.64]
print('The answer is', middle_crime_areas.notnull().sum())

# Lineplot between land area and crime rate
plt.plot(df['land_area'], df['crime_rate'])
plt.xlabel('land area')
plt.ylabel('crime rate')
plt.show()

# Scatterplot between Physicians and hospital beds
plt.figure(figsize=(12, 5))
plt.scatter(df['physicians'], df['land_area'])
plt.xlabel('Physicians')
plt.ylabel('Hospital beds')
plt.show()

# No of people in each of the geographic regions
print(df['region'].value_counts())

# No of people getting each income
plt.hist(df['income'])
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Total income in 1976 in millions of dollars')
plt.show()

# The code works and the project is complete
