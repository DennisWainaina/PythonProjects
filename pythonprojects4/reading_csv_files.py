import numpy as np
import pandas as pd
import matplotlib as plt
import sqlite3
import requests

# We can import data from an external source to be able to read it from this external source for example
# Pandas can read csv files locally or from an external source provided the link has been provided for example:
csv_url = "https://raw.githubusercotent.com/datasets/gdp/master/data/gdp.csv"
df = pd.read_csv(r'C:\Users\user\Desktop\annual-enterprise-survey-2021-financial-year-provisional-csv.csv')
print(df)

# Without pandas there are ways to read data directly from SQL database using the sql database
#  We first establish a connection to the database
conn = sqlite3.connect('chinook.db')
print(conn)

# We then run the cursor instance as an execute method to run against the database
cur = conn.cursor()

# We then use this cursor to get data or to perform operations
# cur.execute('SELECT * FROM employees LIMIT 5;')

# This is used to fetch all the data from a database according to the parameters given
results = cur.fetchall()


# It is also possible to read data from HTML files
# This is by using the requests, library to parse data from an external website for example:

# We first write the name of the website and store it in a variable
html_url = 'https://en.wikipedia.org/wiki/The_Simpsons'

# We then get the website using the requests command
r = requests.get(html_url)

# We then read the data
wiki_tables = pd.read_html(r.text, header=0)
print(len(wiki_tables))

simpsons = wiki_tables[1]
print(simpsons.head())

df = pd.read_excel('Transactions.xlsx')
print(df)
