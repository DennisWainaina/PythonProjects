word = 'book'
print(len(word))

# Program for reversing numbers


def inverse():
    # This is the number to be reversed

    year = '346464747758568588'
    # The reversed number is then placed in a list
    lists = []

    length = len(year)
    # The variable length is used to loop over the entire number

    x = length
    # This variable x is used to print the reverse of the number

    for i in range(length):
        x = x-1
        # Since every number in the variable has an index this reduces that index by 1 for every iteration

        reverses = int(year[x])
        # The number of the specific index is then assigned to a variable and converts it to an integer

        lists.append(reverses)
        # This number is then added to a list

    new = lists

    # The final list is assigned to a variable

    print('The reversed number is ', *new, sep='')
    # The final reversed number without the comma and brackets is then printed


inverse()

unsorted = [5, 4, 8, 6]


year = len(unsorted)
for i in range(year):
    for j in range(0,year-i-1):
        if unsorted[i] > unsorted[i+1]:
            unsorted[i], unsorted[i+1] = unsorted[i+1], unsorted[i]

print(unsorted)

for k in range(year):
    print(k)

print(bin(37))

# Program for printing F shape using Xs
numbers = [5, 2, 5, 2, 2]
for x in numbers:
    output = ''
    for y in range(x):
        output = output + "x"
    print(output)

# Lists are data inside brackets
mylist = ['banana', 'cherry', 'apple']
print(mylist)

# Used to add items to list
mylist.append('lemon')
print(mylist)

# Used to add items to list at a specific index
mylist.insert(1,'blueberry')
print(mylist)
