#Program for displaying Fibonnaci's  sequence of numbers
def fibonaccis():
    a = 1
    b = 0

# The idea is to have two numbers starting from 1 and O which add each other to give a sum
# The sum is then added to b to give 2 numbers one which is the sum of the previous and the other the sum of the current
    num = int(input('How many fibonacci digits do you want to know ? '))

    list = []
    for i in range (num):
        c = a + b
        # This first adds the initial two numbers to give a sum

        d = b + c
        # The sum is then added to an initial number starting from 0

        a = c
        b = d
        z = [a,b]

        list.append(z)
        # Values are then stored in a new variable which is then stored in a list

    list1 = []
    for i in list:
        for t in i:
            list1.append(t)
    # Since values come in lists of 2 we want to seperate them into one list for all the values

    list1.insert(0, 0)
    # We add zero since it was not displayed in the original list

    a = list1[0 : num]
    # We then assign the values of the list to a variable from the beginning to the number of fibonnaci numbers we want

    x = [t for t in a]
    # We then create a list for these values and use a for loop to get the numbers up to the fibonacci no in question.

    for y in x:
        print(y)

    print("These are the fibonacci numbers ")


fibonaccis()

#Program for combining the odd numbers in one list with the even numbers in another list
list1 = [10, 20, 25, 30, 35]
list2 = [40, 45, 60, 75, 90]
# These are the lists containing the numbers to be printed

list3 = []
list4 = []
# list3 is for the odd numbers and list4 is for the even numbers

def mergelist():
    # First we find the odd numbers in list1 and store them in a new list
    for number in list1:
        if (number % 2 ) != 0:
            list3.append(number)



    odd = list3

    # Then we find the even numbers in list 2 and store them in a different list


    for book in list2:
        if (book % 2) == 0:
            list4.append(book)


    even = list4

    # We then combine the two lists to form a new list with the even to odd numbers in the two lists

    new = odd + even
    print('The odd numbers of list 1 and the even numbers of list 2 are',new)

mergelist()


# Program for printing a number starting from the last digit to the first

def inverse():
    reverse = input("What number would you like reversed ? ")
    # First we ask the user for the number he/she would like to reverse

    list = []
    # This list is used to put all the digits of the reversed number gotten from the for loop

    length = len(reverse)
    # The variable length is used to loop over the entire number

    x = length
    # This variable x is used to print the reverse of the number

    for i in range(length):
        # The range is the length of the entire list of numbers

        x = x - 1
        # Since every number in the variable has an index this counts the indexes backwards from the last to the first

        reverses = int(reverse[x])
        # This then stores the number which has that specific index starting from the last to the first

        list.append(reverses)
        # This number is then added to a list

    new = list

    # The final list with all the numbers is assigned to a variable

    print('The reversed number is ', *new, sep='')
    # The final reversed number without the comma and brackets is then printed


inverse()


# Program for calculating income tax
def taxes():
    # The first step is to write the income to be taxed
    income = 75000


    while (income > 0):
        # The while loop is to ensure taxing stops when income = 0
        # We then start taxing the money according to the tax brackets

        if income < 10000:
            print('There is no tax on amounts less than 10000')
            break

        tax = 0*10000

        # We then subtract the first tax bracket from the original income to get new income
        new_income = income - 10000

        if (new_income <= 0):
            print("The tax is 0")
            break

        # From the new income a new tax bracket is calculated for the remaining amount
        tax2 = 0.1*10000

        # The new tax bracket is then subtracted from the remaining amount
        new_income2 = new_income - 10000

        if (new_income2 <= 0):
            print("The tax is ",tax2)
            break

        # The remaining amount is then taxed
        tax3 = 0.2*new_income2

        #All the taxes are then added together to get the total amount taxed
        tax4 = tax + tax2 + tax3

        # The total tax is then printed
        print("The tax for",income,"is", tax4)

        # Then the amount remaining after tax has been deducted
        remaining = income - tax4
        print("The amount remaining after tax is",remaining)

        #This is to close the while loop
        new_income3 = new_income2 - new_income2
        income = new_income3


taxes()


# Program for displaying multiplication tables from 1 to 10

for i in range(1,11):
    for j in range(1,11):
        print(i*j,end=" ")
    print("\t\t")

print('These are the times tables of the numbers 1 to 10')


# Program for printing stars
k = 5
while(k>0):
    print(k*' * ')
    k = k - 1


# Program to find the exponent of a number
def exponents():
    number = int(input("What number would you like to get the exponent of ? "))
    power = int(input("What number would you like to raise your number to ? "))
    exponent = number**power
    print(number,'raised to',power,'is',exponent)

exponents()

print(bin(127))

list1 = [10, 20, 25, 30, 35]
list2 = [40, 45, 60, 75, 90]
x = -1
y = -1

for b in list1:
    x = x + 1
    y = y + 1
    if list1[x] < list2[y]:
        list1[x] = list2[y]
print(list1)
print('Hello world')
print('God is good')
list3 = [5, 6, 7, 8]

def fibonacci():
    a, b = 0, 1
    num = int(input('How many Fibonacci digits do you want to know? '))

    for i in range(num):
        yield a
        a, b = b, a + b

fib = fibonacci()

for n in fib:
    print(n)

print("These are the Fibonacci numbers.")
