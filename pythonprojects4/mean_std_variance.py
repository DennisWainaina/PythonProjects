# Code for finding the mean std and variance of a 3 by 3 numpy array and other parameters of the array
import numpy as np


def calculate(list):
    try:

        B = np.array(list)
        print(B)
        C = np.reshape(B, (3, 3))

        print(C)
        print(C[0])

        first = C[0]

        second = C[1]

        third = C[2]

        column1 = C.T[0]

        column2 = C.T[1]

        column3 = C.T[2]
        print('Axis1 is for the columns of the matrix while Axis2 is for the rows')
        mean_axis2 = [first.mean(), second.mean(), third.mean()]
        mean_axis1 = [column1.mean(), column2.mean(), column3.mean()]
        mean_flattened = C.mean()

        var_axis1 = [column1.var(), column2.var(), column3.var()]
        var_axis2 = [first.var(), second.var(), third.var()]
        var_flattened = C.var()

        std_axis1 = [column1.std(), column2.std(), column3.std()]
        std_axis2 = [first.std(), second.std(), third.max()]
        std_flattened = C.std()

        max_axis1 = [column1.max(), column2.max(), column3.max()]
        max_axis2 = [first.max(), second.max(), third.max()]
        max_flattened = C.max()

        min_axis1 = [column1.min(), column2.min(), column3.min()]
        min_axis2 = [first.min(), second.min(), third.min()]
        min_flattened = C.min()

        sum_axis1 = [column1.sum(), column2.sum(), column3.sum()]
        sum_axis2 = [first.sum(), second.sum(), third.sum()]
        sum_flattened = C.sum()

        print(C.var())
        
        calculations = {
            'mean': [mean_axis1, mean_axis2, mean_flattened],  # Axis1 is for the columns and Axis2 for the rows
            'Variance': [var_axis1, var_axis2, var_flattened],
            'Standard Deviation': [std_axis1, std_axis2, std_flattened],
            'Max': [max_axis1, max_axis2, max_flattened],
            'Min': [min_axis1, min_axis2, min_flattened],
            'Sum': [sum_axis1, sum_axis2, sum_flattened]

        }
        print(calculations)
    except:
        if len(list) != 9:
            raise ValueError('List must contain 9 numbers')


calculate([0, 1, 2, 3, 4, 5, 6, 7, 8])
print('Saudi Arabia beats Argentina in a surprising turn of events')
# The code worked and the project is complete
