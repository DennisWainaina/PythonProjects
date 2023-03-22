import sys
import time


import numpy as np

d = list(range(100001))
e = np.arange(1000)


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
big = sys.getsizeof(5)
print(big)
