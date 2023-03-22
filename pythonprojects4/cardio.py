import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('cardio_base.csv')
print(df)

# How much heavier is the age group with the highest average weight compared to the group with the lowest weight
print(df['age'])
min_max_scaler = MinMaxScaler()
col_name = df['age']
x = df['age']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=col_name)
print(x)
