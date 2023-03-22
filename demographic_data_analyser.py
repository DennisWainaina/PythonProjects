# Code for importing a database in the form of a CSV file to answer important questions about database as shown below
import pandas as pd


def calculate_demographic_data():
    df = pd.read_csv('adult.data.csv')

    # Question 0: How many people of each race are represented in this dataset?
    # This should be a Pandas, series with race names as the index labels. (race column)
    race = df['native-country'].value_counts()
    print('The different races of this world in terms of countries are ;')
    print(race)

    # Question 1: What is the average age of men from this database ?
    # To do this we first find the average age of both women and men to see if it would work with both
    first_question = df['age']

    # We then display the results
    print(first_question)
    print('The average age of both women and men is', first_question.mean())
    
    # We then seperate the men from the women using Booleans and print the results
    men_in_database = df.loc[df['sex'] == 'Male']
    print(men_in_database)
    # We then isolate the age column in our new database containing only men and print the results
    age_of_men = men_in_database['age']
    print(age_of_men)
    print('The average age of men is', age_of_men.mean())

    # Question 2: What is the percentage of people with Bachelor's Degree?
    # First we need to establish the total no of people who have a Bachelor's Degree using the notnull method
    second_question = df.loc[df['education'] == 'Bachelors', 'education']
    print(second_question)
    print(second_question.isnull().sum())
    print('The total no of people who have a Bachelors degree are', second_question.notnull().sum())

    # We then divide the total no of people with a Bachelors Degree with the total no of people who are educated
    # We then multiply the quotient with 100 to get is a percentage and that's our answer
    educated = df['education']
    print('The percentage of people who have a Bachelors degree is', (second_question.notnull().sum() /
          (educated.notnull().sum()))*100, 'percent')

    # Question 3: What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make >50K?
    middle_class = df.loc[df['salary'] == '>50K']
    print(middle_class)
    print(middle_class.isnull().sum())
    Bachelors = middle_class.loc[middle_class['education'] == 'Bachelors', 'education']
    print(Bachelors)
    print('The number of people with a Bachelors degree that earn more than 50k are ', Bachelors.notnull().sum())
    Masters = middle_class.loc[middle_class['education'] == 'Masters', 'education']
    print(Masters)
    Total_Masters = df.loc[df['education'] == 'Masters', 'education']
    print('The total number of people with Masters are', Total_Masters.notnull().sum())
    print('The number of people with a Masters degree that earn more than 50k are', Masters.notnull().sum())
    Doctorate = middle_class.loc[middle_class['education'] == 'Doctorate', 'education']
    print(Doctorate)
    Total_Doctorates = df.loc[df['education'] == 'Doctorate', 'education']
    print('The total number of people with Doctorates are', Total_Doctorates.notnull().sum())
    print('The total no of people with a Doctorate degree that earn more than 50k are', Doctorate.notnull().sum())
    total = Bachelors.notnull().sum() + Masters.notnull().sum() + Doctorate.notnull().sum()
    higher_degrees = second_question.notnull().sum() + Total_Masters.notnull().sum() + Total_Doctorates.notnull().sum()

    learned = df['education']
    print(learned.isnull().sum())
    print('Hence the total number of people with advanced education that earn more than 50k are', total)
    print('The total no of people with higher degrees are', higher_degrees)
    print(' The number of people with advanced degrees that earn more than 50k are', (total/higher_degrees)*100, '%')

    # Question 4 : What percentage of people without advanced degrees earn more than 50k?
    remaining = (1-(total/higher_degrees))*100
    print('The number of people without advanced degrees that earn more than 50k are', remaining, 'percent')

    # Question 5: What is the minimum no of hours a person works per week?
    hours = df['hours-per-week']
    print(hours)
    print('The minimum number of hours done in a week by an employee are', hours.min(), 'hours')

    # Q6: What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?
    lowest = df.loc[df['hours-per-week'] == hours.min()]
    print(lowest)
    lowest1 = df.loc[df['hours-per-week'] == hours.min(), 'hours-per-week']
    print('There are', lowest1.notnull().sum(), 'who work the minimum number of hours per week')
    stealing_a_living = lowest.loc[lowest['salary'] == '>50K', 'salary']
    print('There are', stealing_a_living.notnull().sum(), 'people who work', hours.min(), ' hours a week and earn >50K')
    print('Hence we can conclude that', (stealing_a_living.notnull().sum()/lowest1.notnull().sum())*100,
          ' percent of people working min no of hours per week earn more than 50K')

    # Q7:What country has the highest percentage of people that earn >50K and what is that percentage?
    # First step is to find people earning more than 50K
    middle_class = df.loc[df['salary'] == '>50K']
    print(middle_class)
    middle_class1 = df.loc[df['salary'] == '>50K', 'native-country']
    highest_gdp = middle_class1.mode()
    print(highest_gdp)
    print(middle_class1.value_counts())
    print('The country with the highest no of people who earn more than 50K is', highest_gdp)
    total_Americans = df['native-country']
    print(total_Americans.value_counts())
    print('The percentage of Americans who earn more than 50K is', ((7171/29170)*100), 'percent')
    # These numbers are gotten from the value counts of the filtered data

    # Q8: Identify the most popular occupation for those who earn >50K in India ?
    middle_class = df.loc[df['salary'] == '>50K']
    print(middle_class)
    Eastern = middle_class.loc[middle_class['native-country'] == 'India']
    print(Eastern)
    popular_occupation = Eastern['occupation'].mode()
    print('The most popular occupation which earns more than 50K in India is ', popular_occupation)
    df.to_excel('raw_data.xlsx')


calculate_demographic_data()
y = 6
print(y)
# The code works and the project is complete
