# Program to check if the multiple of two numbers is less than or equals to a 1000
num1 = 40
num2 = 30
if (num1 * num2) <= 1000:
    print(num1 * num2)

else:
    print(num1 + num2)

# Program to print the sum of the current and previous number in a list
for num in range(9):
    sums = num + (num + 1)
    print("Current number is", num + 1, "previous number is", num, "sum is", sums)

# Program to print the even letters in a word
strs = "words"
for use in strs:
    word = strs.index(use)
    if word % 2 == 0:
        print(use)


# Program to check if the first and last numbers in the list are the same


def same():
    x = [10, 20, 30, 40, 20]

    if x[0] == x[-1]:
        return True
    else:
        return False


print("Result is", same())


