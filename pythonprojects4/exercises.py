
# Program for determining the year a person will be 100 years old


def centenarian():
    name = input("What is your name ? ")
    age = int(input('What is your age ? '))
    current_year = int(input('What is the current year ? '))
    old = 100
    difference = old - age
    new_year = current_year + difference
    print(name, 'you will be 100 years in', new_year)


centenarian()

# Program for determining whether a number is odd or even


def even():
    number = int(input('What number would you like to know whether it is odd or even ? '))
    if number % 2 == 0:
        print(number, 'is even')
    else:
        print(number, 'is odd')


even()

# Program for printing numbers less than 5 in  a list
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
lists = []
for i in a:
    if i < 5:
        lists.append(i)
print(lists)

# Program for printing the Fibonacci sequence of numbers


def fibonacci():
    a = 1
    b = 0
# The idea is to have two numbers starting from 1 and O which add each other to give a sum
# The sum is then added to b to give 2 numbers one which is the sum of the previous and the other the sum of the current

    for i in range(1, 7):
        c = a + b
        # This first adds the initial two numbers to give a sum

        d = b + c
        # The sum is then added to b

        a = c
        b = d
        z = a, b
        print(*z, end=" ")

    print("These are the fibonacci numbers ")


fibonacci()

# Program for writing the words of a sentence in reverse


def reversed_word():
    name = 'My name is Margaret'
    splitted = name.split()

    y = len(splitted)
    list1 = []
    for split in splitted:
        y = y - 1
        list1.append(splitted[y])
    print(*list1)

    
reversed_word()
