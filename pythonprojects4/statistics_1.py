# How to calculate variance
import numpy as np


def variance():
    # First you get the mean of the group numbers by adding them all together and dividing them by the total no of no.
    list1 = [55, 25, 35, 75, 21, 33, 37, 25, 15, 18]
    a = 0
    for i in list1:
        a = a + i
    print(a)
    mean = a/len(list1)
    print('The mean of the group of number is', mean)
    # Then you get the difference of these numbers from the mean then you square them
    list2 = []
    for i in list1:
        t = i - mean
        t = t**2
        list2.append(t)
    print('The squares of the difference of numbers from mean is', list2)
    # You get the mean of these squares and that is your variance
    b = 0
    for r in list2:
        b = b + r
    print(b)
    mean1 = b/len(list2)
    print('The variance is', mean1)


variance()
first = np.array([5, 17, 23, 31, 43, 49, 57, 17, 57, 17])
print('The mean of the array is', first.mean())
first.sort()
print(first)

# Probability is the chance of something happening
# Probabilty distribution is the chance of a certain variable occuring
# In the P.D graph the x-axis represents the variable while the y-axis represents the chance that value variable occurs
# Machine learning ability of machine to learn and imporve from experience without being programmed
print(x for x in first)
