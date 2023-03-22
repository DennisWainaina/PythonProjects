import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Question 1:Set index to the date column
df = pd.read_csv('fcc-forum-pageviews.csv')
# First we see how the dataframe looks like before coding
print(df)
# We then set the index to the date column
df.index = df['date']
print(df.index)

# We see the effect that has to the dataframe
print('The dataframe before the column date has been dropped')
print(df)
df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)

print(df)

# We then drop the date column since there are two columns with data on them and print to see what happened
df.drop('date', inplace=True, axis=1)
print('The dataframe after the column date date has been dropped')
print(df)

# We then save the results to an Excel file to see if it worked
df.to_excel('time_series_values.xlsx')


# Question 2: Filter out data by removing days which were in the top 2.5% or bottom 2.5% of the dataset
def filtered():
    value = 'value'

    correct_date = df[value][((df[value]) >= (df[value].quantile(0.025))) &
                            ((df[value]) <= (df[value].quantile(0.975)))]
    correct_date = df['value']
    print(correct_date)
    print(df)
    df.to_excel('filtered_time_series_values.xlsx')


filtered()


# Question 3: Create a draw_line_plot function that uses Matplotlib to draw a line chart
def draw_line_plot():

    fig, ax = plt.subplots(figsize=(32, 10), dpi=100)
    plt.ylim(20000, 180000)
    plt.xlim(100, 1000)
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    plt.xlabel('Date')
    plt.ylabel('Page Views')
    plt.plot(df.index, df['value'], 'r', linewidth=1)
    plt.show()

    fig.savefig('line_plot.png')
    return fig


draw_line_plot()


def draw_bar_plot():

    plt.bar(x=df.index, height=df['value'])
    plt.ylim(20000, 180000)
    plt.xlim(100, 1000)
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    plt.xlabel('Date')
    plt.ylabel('Page Views')

    plt.show()
    print(df.index[0])


draw_bar_plot()


def draw_box_plot():
    fig, a = plt.subplots(1, 2, figsize=(16, 5))

    f1 = fig.add_subplot(111)
    f2 = fig.add_subplot(122)

    f1.set_title('Year-wise Box Plot (Trend)')
    plt.ylim(20000, 180000)
    f1 = sns.boxplot(x=df.index, y=df['value'], data=df)

    f2.set_title('Month-wise Box Plot (Seasonality)')
    f2 = sns.boxplot(x=df['month'], y=df['value'], data=df)
    f2.set_xlabel('Month')
    f2.set_ylabel('Page views')
    plt.ylim(20000, 180000)
    plt.show()


draw_box_plot()
print('Man U fires Ronaldo in a suprising turn of events')
# The code works and the project is complete
