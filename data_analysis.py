import sys
import time


import numpy as np
# This is a module used to analyse data in form of arrays
# It is faster than using the normal arrays in Python as it uses less storage to process the data than Python

a = np.array([1, 2, 3, 4])
b = np.array([0.5, 1, 2, 2.5])
# In order to use numpy arrays one creates arrays like normal but beginning with numpy

A = np.array(
    [[1, 2, 3],  # This is row 0 hence for example if we wanted 2 from the array we'd say [0][1]
     [4, 5, 6],  # This is row 1
     [7, 8, 9]  # This is row 2
     ])
# We can have one or multiple dimensional arrays like the one above is two_dimensional

print(A.shape)
# This shows the number of rows and columns in that order

print(A[2, 2])
# This shows the position of the element in the 2nd row and 2nd column which is 9

print(A.ndim)
# This shows the total no of dimensions present in the array for example A has 2 dimensions

B = np.array([
    [
        [12, 11, 10],
        [9, 8, 7]

    ],
    [
        [6, 5, 4],
        [3, 2, 1]
    ]
])
print(B[0, 1, 2])


# Vectorised operations are operations performed between arrays and arrays and arrays and scalars
# For example
ray1 = np.arange(4)
print(ray1)
ray2 = ray1 + 10
print(ray2)
print(ray1)

# The above examples are between arrays and scalars we can even multiply or divide them
# We can also perform operations between arrays for example;
ray3 = np.array([10, 10, 10, 10])
ray4 = ray3 + ray1
print(ray4)

ray5 = np.arange(5)
ray6 = ray5 + 20
print(ray6)

print(ray1 >= 2)
# One can also use boolean operations to filter out unnecessary data according to a certain like <=2
# If one prints the boolean condition it will return true or false depending on the condition given
# One can also print arrays based on the condition given by the boolean array for example;

print(ray1[ray1 >= 2])
# The code 65 above prints all the numbers in the list greater than or equals to 2 hence filtering out the data

C = np.array([
    [4, 5],
    [6, 7],
    [8, 9]

])
print(C[1, 1])

# We now want to look at the difference in size of data between numpy and python
print(sys.getsizeof(1))
# As can be seen from line 77 it takes 28 bytes to store one integer in python
# Note 1 byte is equals to 8 bits
print(sys.getsizeof(A))

# We can compare the time it takes python to perform operations on normal python lists versus numpy arrays
# For example;
d = list(range(1000))
e = np.arange(1000)

# Program for printing the squares of numbers from 1 to 1000 and putting them in one list
# We can measure the time taken for the program to execute using the time module
# This is by subtracting the difference between the starting and ending time of the function in question

start = time.time()


def squares():
    lists = []
    for i in range(1, 1001):
        i = i ** 2
        lists.append(i)
    print(lists)


squares()
end = time.time()
# We then print the time taken for the function to execute
print('The time taken for the squares function to take execute is ', end-start, 'seconds')


def squares1():
    start = time.time()
    lists1 = []
    for t in e:
        t = t ** 2
        lists1.append(t)
    print(lists1)
    end = time.time()
    # We then print the time taken for the function to execute
    print('The time taken for  the numpy array to  execute is ', end-start, 'seconds')


squares1()


def squares2():
    start = time.time()
    lists2 = []
    for x in d:
        x = x ** 2
        lists2.append(x)

    print(lists2)
    end = time.time()
    # We then print the time taken for the function to execute
    print('The time taken for the normal list to  execute is ', end-start, 'seconds')


squares2()
# As seen from the results numpy arrays take less time to execute functions
print(A.mean())
print(A.mean(axis=0))
print(B.sum(axis=0))
