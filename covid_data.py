import pandas as pd
import numpy as np

df = pd.read_csv('covid_data.csv')
print(df)
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
categorical_variables = list(set(df.columns) - set(numerical_variables))
print(numerical_variables)
print(categorical_variables)
# Filling null values
df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].mean())
df[categorical_variables] = df[categorical_variables].fillna(df[categorical_variables].mode().iloc[0])
print(df)  # No more null values

# Probability country has GDP over 10,000 if 5 hospital beds per 1000 inhabitants
five_hospital_beds = df['hospital_beds_per_thousand'][((df['hospital_beds_per_thousand']) >= 5) &
                                                      ((df['gdp_per_capita']) > 10000)]
print('Those with 5 hospital beds and GDP greater than 10000 are', five_hospital_beds)
print('The total number of people with 5 hospital beds and GDP greater than 10000 are', len(five_hospital_beds))
good_countries = five_hospital_beds[df['gdp_per_capita'] > 10000]
print(len(good_countries))
print('The total no of people are', len(df['hospital_beds_per_thousand']))
probability = len(good_countries) / len(df['gdp_per_capita'])
print(probability)
good_economies = df['gdp_per_capita'][df['gdp_per_capita'] > 10000]
good_healthcare = good_economies[df['hospital_beds_per_thousand'] >= 5]
print(len(good_healthcare))
print(len(df['gdp_per_capita']))
print(3236/23082)

# Country with the third-highest death rate
print(df['new_deaths'].value_counts())
Spain = df['location'][df['new_deaths'] == df['new_deaths'].max()]
print(Spain)
USA = df['new_deaths'][df['location'] == 'United States']
print(USA.max())
Italy = df['new_deaths'][df['location'] == 'Italy']
print(Italy.max())
Spain = df['new_deaths'][df['location'] == 'Spain']
print(Spain.max())
Belgium = df['new_deaths'][df['location'] == 'Belgium']
print(Belgium.max())
Andorra = df['new_deaths'][df['location'] == 'Andorra']
print(Andorra.max())
countries = df['location'].unique().tolist()
print(countries)
Afghanistan = df['new_deaths'][df['location'] == 'Afghanistan']
Albania = df['new_deaths'][df['location'] == 'Albania']
Algeria = df['new_deaths'][df['location'] == 'Algeria']
Andorra = df['new_deaths'][df['location'] == 'Andorra']
UK = df['new_deaths'][df['location'] == 'United Kingdom']
print(Afghanistan)
i = 0
list_of_countries = []
for t in Afghanistan:
    i = i + t
print('The total no of deaths in Afghanistan is', i)
for t in Albania:
    i = i + t
print('The total no of deaths in Albania is', i)
for t in Algeria:
    i = i + t
print('The total no of deaths in Algeria is', i)
for t in Andorra:
    i = i + t
print('The total no of deaths in Andorra is', i)
for t in UK:
    i = i + t
print('The total no of deaths in UK is', i)
new_deaths = df['new_deaths'][df['new_deaths'] != 0]
print(new_deaths)
print(new_deaths.value_counts())
new_countries = df['location'][df['new_deaths'] != 0]
print(new_countries)
print(new_countries.value_counts())
Burundi = df['new_deaths'][df['location'] == 'Burundi']
Iran = df['new_deaths'][df['location'] == 'Italy']
i = 0
list_of_countries = []
for t in Italy:
    i = i + t
print(i)
World = df['new_deaths'][df['location'] == 'World']
i = 0
list_of_countries = []
for t in World:
    i = i + t
print(i)
for t in Iran:
    i = i + t
print(i)
deaths = []

list_of_countries.sort(reverse=True)
print(list_of_countries)
print(len(list_of_countries))
total = []
for t in countries:
    l = df['new_deaths'][df['location'] == t]
    x = list(l)
    p = sum(x)
    deaths.append(l)
    total.append(p)
print('The total of the numbers in a list is', total)
print(deaths)
print('The no of items in the list is', len(deaths))

dictionary = {}
for i in range(len(countries)):
    dictionary[countries[i]] = total[i]
print(dictionary)
countries_and_deaths = pd.DataFrame([dictionary])
print(countries_and_deaths)
total.sort(reverse=True)
print(total)

keys = list(dictionary.keys())
vals = list(dictionary.values())
print('The country with the third highest death rate is', keys[vals.index(40883)])
