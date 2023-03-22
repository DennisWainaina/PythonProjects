import datetime


def main_function(input):
    # Place your code here
    x = str(input)
    list_of_numbers = [int(i) for i in x]
    for t in range(1):
        if len(list_of_numbers) > 2:
            break
        if len(list_of_numbers) < 2:
            print(input)
            break
        else:
            y = list_of_numbers[0] * list_of_numbers[1]
            print(y)
        print(len(list_of_numbers))


main_function(3)

t = [5, 6, 7, 8]
r = 'book'
print(len(r))

# Code for converting binary to decimal


def main_function(input):

    # Place your code here
    x = str(input)
    list_of_numbers = [int(i) for i in x]
    y = len(x)
    r = -1
    list1 = []
    for i in range(len(x)):
        y = y - 1
        r = r + 1
        z = 2**r
        t = list_of_numbers[y] * z
        list1.append(t)
    print(sum(list1))


main_function(10100)


def func(name, age=37):
    print(name,age)


func('Pascal', 50)

if True:
    while True:
        print(True)
        break

count = 0
number = 180
while number > 10:
    count = count + 1
    number = number / 3
print("Total no of iterations is", count)


def person(name, age):
    x = ('Portia', 9)
    if name == 'Alice':
        print("Hi Alice")


# Program for converting AM and PM to 24 hr time format


def timeConversion(s):
    try:
        time_obj = datetime.datetime.strptime(s, '%I:%M:%S%p')
        return time_obj.strftime('%H:%M:%S')
    except ValueError:
        return 'Invalid time format'


result = timeConversion('12:06:45AM')
print(result)
