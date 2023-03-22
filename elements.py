from math import factorial
import numpy as np
x = 2
x = x + 2
print(x)
astr = 'you'


for i in [2, 1, 5]:
    print(i)

smallest = None
print("Before:", smallest)
word = "bananana"
stuff = 'Hello World'
t = word.find("na")
print(dir(stuff))

# We can use the open command, so it is possible to read the file not read the file
fhand = open('LICENSE.txt')
print(fhand)

# Program for printing a sentence from the last word to the first word in the sentence
name = 'My name is Dennis'
splitted = name.split()
print(splitted)
print(splitted[3])
y = len(splitted)
list1 = []
for i in splitted:
    y = y-1
    list1.append(splitted[y])
print(*list1)

# One of the methods which one can read the file is using a for loop which prints each line for example:
for sentence in fhand:
    print(sentence)
# This prints each sentence in the line

# We can also count the number of lines in the file for example:
fhand = open('LICENSE.txt')
count = 0
for book in fhand:
    count = count + 1
print('The number of lines is ', count)
# Note that the code doesn't work if the file is read only.

# One can deternmine the number of characters in the file using the read command for example:
fhand = open('LICENSE.txt')
inp = fhand.read()
print('The no of characters in the line are', len(inp))

# We can also print lines which meet a certain criteria using the startswith method for example:
fhand = open('LICENSE.txt')
for line in fhand:
    if line.startswith('the'):
        print(line)

# One can also remove the whitespace after each line using the rstrip method for example:
fhand = open('LICENSE.txt')
for line in fhand:
    line = line.rstrip()
    if line.startswith('the'):
        print(line)

# Testing what the split command does for indiviual strings in various indexes
words = 'His e-mail is q-lar@freecodecamp.org'
pieces = words.split()
print(pieces)
parts = pieces[3].split('-')
print(parts)
n = parts[1]
print(n)
print(pieces)
a = 'Hello World'
print(a[-4])
print(a[-5:-1])
print(a.upper())
x = 36 / 4 * (3 + 2) * 4 + 2
print(x)
y = 2*2**3
print(y)

numbers = [4, 3, 1, 2]
numbers.sort()
print(numbers)

aTuple = 'Yellow', 20, 'Red'
a, b, c = aTuple
print(a)

aTuple = "Orange"
print(type(aTuple))

lst = [3, 4, 6, 1, 2]
lst[1:2] = [7, 8]
print(lst)


for num in range(10, 14):
    for i in range(2, num):
        if num % i == 1:
            print(num)
            break


d = {'john': 40, 'peter': 45}
print(list(d.keys()))
print(len(d))


print('The factorial of 5 is', factorial(5))


A = [[-2, 5, 6],
     [5, 2, 7]]
print(A[0][2])
print(A[0][0])

# Converting array to list
a = np.array([1, 2, 3, 5, 8])
print(a.ndim)

# Converting array to list
print(list(a))

# Returns no of elements in  a list
print(np.size(a))

a = np.zeros(6)
b = np.arange(0, 10, 2)

data = ['peter', 'paul', 'MARY', 'gUIDO']
for s in data:
    s.capitalize()
    print(s)

# Program for finding the factorial of a number


def factorial():
    y = 1
    i = 1
    n = int(input('What number would you like to get the factorial of ? '))
    for k in range(1, n):
        y = y + 1
        i = i * y

    print('The factorial of', n, 'is', i)


factorial()

print('The times table of 2 is')
for t in range(1, 11):
    i = 2
    i = i*t
    print(i, end=" ")
my_list = [1, 2, 3, 4, 5, 6]
print(my_list[3])
name = 'My-name-is-Dennis'
name.split('-')
print(name)
print([1, 2, 3]*3)
x ={1, 2, 3, 4, 5, 6}
x.add(5)
x.add(6)
x.add(7)
print(x)
a = np.array([1, 2, 3, 4])
print(a[[False, True]])
number = 3
print(f'The number is{number}')
