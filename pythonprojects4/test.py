from itertools import product
import random as rand


def print_nums(x):
    for i in range(x):
        print(i)
        return


print_nums(10)


def func(x):
    res = 0
    for i in range(x):
        res = res + i
    return res


print(func(4))

primes = {1: 2, 2: 3, 4: 7, 7: 18}
print(primes[primes[4]])

# Dictionary
fib = {1: 1, 2: 1, 3: 2, 4: 3}
print(fib.get(5, 0))
print(fib)

# Step sizes
squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares[7: 4: -1])

# Formatting
print("{0}{1}{1}".format('abra', 'cad'))

# Program for determining if a number is a prime number (shorter version)
n = int(input('Which number would you like to know is a prime number ? '))

a = [i for i in range(2, n) if n % i == 0]

print('The multiples of', n, 'are', a)

if not a:
    print(n, 'is a prime number')
else:
    print(n, 'is not a prime number')


def my_func(f, arg):
    return f(arg)


print(my_func(lambda x: 2*x*x, 6))


def make_word():
  word = ""
  for ch in "spam":
    word += ch
    yield word


print(list(make_word()))

first = 'book'
second = first.upper()
print(second)


def fib(x):
    if x == 0 or x == 1:
     return 1
    else:
     return fib(x-1) + fib(x-2)


print(fib(4))

a = {1, 2}
print((list(product(range(3), a))))


def power(x, y):
    if y == 0:
        return 1
    else:
        return x * power(x, y - 1)


print(power(2, 3))


# Program for guessing a number and seeing if the number is correct
# It also subtracts points from 10 and stops if points are at 0 and below
# Such that is stops if the number of tries is 0 or less
random_number = rand.randint(1, 100)
# print('The random number is', random_number)
# print(random_number)
b = [i for i in range(2, random_number) if random_number % i == 0]


tries = 10

while tries > 0:
    guessed_number = int(input('Guess the random number from 1 to 100 ? '))

    if random_number == guessed_number:
        print('You are correct')
        print('You guessed correctly')
        print('The final number of points you have is', tries)
        break

    else:
        print('Not the correct number')
        print('The random number is a multiple of', b)
        print('The number of points you have is', tries - 1)
        print("You're almost there")

    tries = tries - 1
else:
    print('The random number is', random_number)
    print('You failed the guesssing game')
