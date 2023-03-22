import datetime


# Code for finding the maximum and minimum of the sum of 5 elements in a list


def minmax():
    # We first define the elements of the list
    arr = [5, 5, 5, 5, 5]

    # This is a condition to check if all the numbers of the list are the same as the code doesn't work if the same.
    if all(x == arr[0] for x in arr):
        print(arr[0] * 4, arr[0] * 4)
        return

    # The idea is to skip one element and sum the rest beginning with the first element in the list.
    x1 = 0
    for i in arr:
        if i == arr[4]:
            pass
        else:
            x1 = x1 + i
    # We then print the result
    print(x1)

    # We do the same where we skip the second element
    x2 = 0
    for i in arr:
        if i == arr[3]:
            pass
        else:
            x2 = x2 + i

    print(x2)

    # Then the third element
    x3 = 0
    for i in arr:
        if i == arr[2]:
            pass
        else:
            x3 = x3 + i
    print(x3)

    # Then the fourth element
    x4 = 0
    for i in arr:
        if i == arr[1]:
            pass
        else:
            x4 = x4 + i
    print(x4)

    # And finally the fifth element
    x5 = 0
    for i in arr:
        if i == arr[0]:
            pass
        else:
            x5 = x5 + i

    # We then put the results in a list and print the maximum and minimum of the list.
    numbers = [x1, x2, x3, x4, x5]
    print(min(numbers), max(numbers))


minmax()


# Code for converting time format from 12hr format to 24 hr format


def timeConversion(s):
    try:
        time_obj = datetime.datetime.strptime(s, '%I:%M:%S%p')
        return time_obj.strftime('%H:%M:%S')
    except ValueError:
        return 'Invalid time format'


result = timeConversion('12:06:00AM')
print(result)

# Code for comparing two elements in a list and counting how many times they appear
random_items = ['book', 'book', 'year']
queries = ['book', 'pen', 'year']
final = []
for i in queries:
    count = random_items.count(i)
    final.append(count)
print(final)

# Code for checking unique items in a list
a = [1, 2, 3, 4, 3, 2, 1]
n = len(a)

# Sort the array
a.sort()

# Check for first element
if a[0] != a[1]:
    print(a[0], end=" ")

# Check for all the elements
# if it is different its
# adjacent elements
for i in range(1, n - 1):
    if (a[i] != a[i + 1] and
            a[i] != a[i - 1]):
        print(a[i], end=" ")

# Check for the last element
if a[n - 2] != a[n - 1]:
    print(a[n - 1], end=" ")


def book(cover):
    year = cover + 2
    print(year)


book(6)

# Code for total cost in a restaurant


def solve(meal_cost, tip_percent, tax_percent):
    if not isinstance(meal_cost, (int, float)) or not isinstance(tip_percent, int) or not isinstance(tax_percent, int):
        print("Error: Input values must be numeric.")
        return
    if meal_cost < 0:
        print("Error: Meal cost must be greater than or equal to 0.")
        return
    if tip_percent < 0 or tip_percent > 100:
        print("Error: Tip percent must be between 0 and 100.")
        return
    if tax_percent < 0 or tax_percent > 100:
        print("Error: Tax percent must be between 0 and 100.")
        return
    total_cost = meal_cost + (0.01 * tip_percent * meal_cost) + (0.01 * tax_percent * meal_cost)
    print(round(total_cost))


if __name__ == '__main__':
    meal_cost = float(input().strip())

    tip_percent = int(input().strip())

    tax_percent = int(input().strip())

    solve(meal_cost, tip_percent, tax_percent)


# Coding challenge known as flipping bits
n = int(input('Enter your number: '))
n = str(n)
numbers = []
for i in range(31):
    n = '0' + n
    numbers.append(n)
print(numbers[30])
x = numbers[30]
a = -1
t = 33
finals = []
for i in range(len(x)):
    x = int(x)
    a = a + 1
    r = 2 ** a
    y = i ** r
    finals.append(y)
print(finals)
