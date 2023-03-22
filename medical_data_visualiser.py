import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

crash_df = sns.load_dataset('car_crashes')
print(crash_df)
df = pd.read_csv('medical_examination.csv')
print(df.head)
height = df['height']
print(height)
print('The average height is', height.mean())

# Q1 : Creating an overweight column such that 1 is overweight and 0 is not overweight


def overweighted():
    height_in_metres = df['height']/100
    print(height_in_metres)
    df['BMI'] = df['weight'] / height_in_metres**2
    print(df['BMI'])
    print(df)
    bmi = df['BMI']
    print(bmi)
    list1 = []
    for i in bmi:
        if i > 25:
            overweight = 1
            list1.append(overweight)
        else:
            not_overweight = 0
            list1.append(not_overweight)

    df['overweight'] = list1

    print(df)
    

overweighted()

# Q2: Normalise the data If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1,
# make value 1


def normalise():
    cholestrol = df['cholesterol']
    list2 = []

    for t in cholestrol:
        if t > 1:
            t = 1
            list2.append(t)
        else:
            t = 0
            list2.append(t)

    df['cholesterol'] = list2
    print(df['cholesterol'])
    print(df)
    # This is to check if the code actually worked

    glucose = df['gluc']
    list3 = []
    for g in glucose:
        if g > 1:
            g = 1
            list3.append(g)
        else:
            g = 0
            list3.append(g)

    df['gluc'] = list3
    print(df['gluc'])
    print(df)
    # We do the same for the gluc column


normalise()


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco',
    # 'active', and 'overweight'.
    df1 = df[["cardio", "active", "alco", "cholesterol", "gluc", "overweight", "smoke"]]
    df_cat = pd.melt(df1, id_vars='cardio')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    #  You will have to rename one of the columns for the catplot to work correctly.
    # df_cat = None

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x="variable", col="cardio", hue="value", data=df_cat, kind="count").set_axis_labels("variable",
                                                                                                          "total").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


draw_cat_plot()


# Cleaning the data


def cleaning():
    # The first step is to filter out the data in ap_lo which is greater than the ap_hi column
    corrected_values = df['ap_lo'][df['ap_lo'] <= df['ap_hi']]
    print(corrected_values)
    corrected_values = df['ap_lo']
    print(df['ap_lo'])
    print(df)

    # The next step is to filter out data according to a certain percentile
    correct_height = df['height'][((df['height']) >= (df['height'].quantile(0.025))) &
                                  ((df['height']) < (df['height'].quantile(0.975)))]
    print(correct_height)
    df['height'] = correct_height
    correct_height1 = df['height'][(df['height']) < (df['height'].quantile(0.975))]
    print(correct_height1)

    correct_weight = df['weight'][((df['weight']) < (df['weight'].quantile(0.975))) &
                                  ((df['weight']) > (df['weight'].quantile(0.025)))]
    print(correct_weight)
    df['weight'] = correct_weight
    correct_weight1 = df['weight'][(df['weight']) > (df['weight'].quantile(0.025))]
    print(correct_weight1)

    print(df)


print('This is the final piece of code')

df.to_excel('medical_data.xlsx')

cleaning()


def draw_heat_map():
    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=1.4)

    df_heat = df

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, square=True, annot=True, fmt="0.1f", vmax=.32, cmap='gnuplot2', linewidths=1)
    plt.show()

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig


draw_heat_map()
print('Japan beats Germany in a surprising turn of events')
# The code works and the project is complete
