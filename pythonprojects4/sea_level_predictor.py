import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


df = pd.read_csv('epa-sea-level.csv')
print(df)


def draw_plot():
    # Creating a scatterplot of the data from the CSV
    fig, ax = plt.subplots(figsize=(32, 10))
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], marker='x')
    plt.xlabel('Year')
    plt.ylabel('Sea Level in Inches')
    plt.title('Rise in Sea Level')

    # Creating first line of bestfit from the scatterplot upto the year 2050
    regression_data = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    print(regression_data)

    x_fit1 = list(range(1880, 2051))
    list1 = []
    for i in x_fit1:
        y_fit = ((regression_data[0]) * i) + regression_data[1]
        list1.append(y_fit)

    ax.plot(x_fit1, list1)

    # Creating second line of bestfit form 2000 to 2050
    x_fit2 = list(range(2000, 2051))

    list2 = []
    for y in x_fit2:
        y_fit = ((regression_data[0]) * y) + regression_data[1]
        list2.append(y_fit)

    ax.plot(x_fit2, list2)

    plt.show()
    df.to_excel("epa_sea_level.xlsx")

    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plots.png')
    return fig


draw_plot()
print('Richarlison scores one of the goals of the season in the World Cup')
# The code works and the project is complete
