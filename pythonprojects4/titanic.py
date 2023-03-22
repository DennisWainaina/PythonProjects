import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/First_ML_Model/master/titanic.csv')
print(df)
print(df.info())
print(df.head())

# Whether fare has missing values and if males are more than women
print('The missing values in the fare column are', df['Fare'].isnull().sum())
print(df['Sex'].notnull().sum())
male = df['Sex'].loc[df['Sex'] == 'male']
print('The no of male passengers are', male.notnull().sum())
female = df['Sex'].loc[df['Sex'] == 'female']
print('The no of female passengers are', female.notnull().sum())

# The ratio of people who survived the crash of the titanic
survivors = df['Survived'][df['Survived'] == 1]
print(survivors)
print('The no of people who survived are', survivors.notnull().sum())
total_passengers = df['Survived'].notnull().sum()
print(df['Survived'].isnull().sum())
print('The total no of passengers in the ship are', total_passengers)
ratio = survivors.notnull().sum() / total_passengers
print('The ratio of the survivors to the total passengers is', ratio)

# What is the median fare of the passengers?
median_fare = df['Fare'].median()
print('The median fare of the passengers is', median_fare)

# Percentage of women that survived
female = df['Sex'].loc[df['Sex'] == 'female']
print(female)
women_survived = female.loc[df['Survived'] == 1]
print('The missing values in the women that survived column is', women_survived.isnull().sum())
print('The total no of women who survived the crash are', women_survived.notnull().sum())
print('The missing value in the column that contains women only are', female.isnull().sum())
print('The total no of women who survived are', female.notnull().sum())
print('The percentage of women who survived is therefore', (233/314)*100)
print('The percentage of men who survived is', (100-74.2), 'percent')
print('Therefore the percentage of women who survived is higher')

# Percentage of first class passengers who survived
first_class = df['Pclass'].loc[df['Pclass'] == 1]
print(first_class)
first_class_survivors = first_class.loc[df['Survived'] == 1]
print('The no of first class passengers who survived are', first_class.notnull().sum())
print('The no of first class passengers who survived are', first_class_survivors.notnull().sum())
print('The percentage of first class passengers who survived is therefore', (136/216)*100)
print('Therefore it seems that first class passengers were given priority')

# Percentage of second class passengers who survived
second_class = df['Pclass'].loc[df['Pclass'] == 2]
print(second_class)
second_class_survivors = second_class.loc[df['Survived'] == 1]
print('The no of second class passengers is', second_class.notnull().sum())
print('The no of second class passengers who survived are', second_class_survivors.notnull().sum())
print('Therefore the percentage of second class passengers who survived were', (87/184)*100)

# Percentage of third class passengers who survived
third_class = df['Pclass'].loc[df['Pclass'] == 3]
print(third_class)
third_class_survivors = third_class.loc[df['Survived'] == 1]
print('The no of third class passengers is', third_class.notnull().sum())
print('The no of third class passengers who survived are', third_class_survivors.notnull().sum())
print('The percentage of third class passengers who survived is therefore', (119/491)*100)

# Percentage of children who survived
print(df['Parch'])
print(df['Age'])
corrected_values = df['Age'].fillna(df['Age'].mean())
print('The corrected values are')
print(corrected_values)
df['Age'] = corrected_values
print('The no of null values in the age column is', df['Age'].isnull().sum())
print(df['Age'])
children = df['Age'].loc[df['Age'] < 18]
print('The no of null values in the age column less than 18 is', children.isnull().sum())
print('The number of children who were in the titanic were', children.notnull().sum())
surviving_childen = children[df['Survived'] == 1]
print(surviving_childen)
print('The number of children who survived were', surviving_childen.notnull().sum())
print('Therefore the percentage of children who survived were', (61/113)*100)
print('The percentage of people who are above 18 that survived is', (100-(61/113)*100))

# No of survivors who embarked from Southampton
print('The no of null values in the embarked column are', df['Embarked'].isnull().sum())
new_values = df['Embarked'].fillna(method='ffill', inplace=True)
print('After filling the null values the no of null values is', df['Embarked'].isnull().sum())
Southamptoners = df['Embarked'].loc[df['Embarked'] == 'S']
print(Southamptoners)
print('The no of of people who boarded at Southampton are', Southamptoners.notnull().sum())
survived_Southamptoners = Southamptoners.loc[df['Survived'] == 1]
print(survived_Southamptoners)
print('The no of people who survived from Southampton are', survived_Southamptoners.notnull().sum())

# What are the five highest fares ?
print(df['Fare'].isnull().sum())
print(df['Fare'].value_counts())
print(df['Fare'].loc[df['Fare'] >= 500])
print(df['Fare'].loc[(df['Fare'] >= 500) & (df['Fare'] <= 400)])
print(df['Fare'].loc[(df['Fare'] >= 300) & (df['Fare'] <= 400)])
print(df['Fare'].loc[(df['Fare'] >= 250) & (df['Fare'] <= 300)])

# What is the median age of the passengers ?
print('The median age of the passengers is', df['Age'].median())

# No of unique values in the name column
unique_names = df['Name'].loc[df['Name'] == df['Name'].unique()]
print(unique_names)

# Most passengers have how many siblings and spouses ?
print('The no of passengers who have a specific number of siblings are', df['SibSp'].value_counts())
df.to_excel('Titanic.xlsx')

# Factor which determines the chances of survival of a passenger
print('The number of null values in the name column are', df['Name'].isnull().sum())
survived_names = df['Name'][df['Survived'] == 1]
print('The no of people in the name column that surived are', survived_names)
print(df['Name'])
print('The percentage of people in the names column who survived were', (342/891)*100)
print('The number of null values in the age column are', df['Age'].isnull().sum())
survived_age = df['Age'][df['Survived'] == 1]
print(survived_age)
print(df['Age'])
print('The percentage of people in the age column who survived were', (342/891)*100)
print('The number of null values in the ticket column are', df['Ticket'].isnull().sum())
survived_tickets = df['Ticket'][df['Survived'] == 1]
print(survived_tickets)
print(df['Survived'].loc[df['Survived'] == 1])
survived_passengers = df['Survived'][df['Survived'] == 1]
print(survived_passengers.notnull().sum())
print('The no of null values in the survived column is', df['Survived'].isnull().sum())
matrix = np.corrcoef(corrected_values, df['Survived'])
print(matrix)
print(df['Age'].loc[(df['Age'] <= 18)])
print(df['Ticket'].value_counts())
highest_number = df['Ticket'][df['Ticket'] == '3101295']
print(highest_number.notnull().sum())
survived_specific_tickets = highest_number[df['Survived'] == 1]
print(survived_specific_tickets.notnull().sum())
matrix = df.corr()
print('The correlation matrix is shown below')
print(matrix)
print(df.columns)
# The code works and the project is complete
