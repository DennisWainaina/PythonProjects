# Program for guessing a number and seeing if the number is correct
# It also subtracts points from 10 and stops if points are at 0 and below
# Such that is stops if the number of tries is 0 or less
import random as rand

random_number = rand.randint(1, 100)
# print('The random number is', random_number)
# print(random_number)
b = [i for i in range(2, random_number) if random_number % i == 0]
if b:
    print('Print the number is divisible by', b)

else:
    print('This is a prime number and is only divisible by 1 and itself')


tries = 10


while tries > 0:
    guessed_number = int(input('Guess the random number from 1 to 100 ? '))

    if random_number == guessed_number:
        print('You are correct')
        print('You guessed correctly')
        print('The number of points you have is', tries)
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

# Binary search algorithm for searching for a number within a list
random_numbers = list(range(1, 100))
rand.shuffle(random_numbers)
# print(random_numbers)
first_number = random_numbers[0]
print('The first number in the list is', first_number)
first_group = []

for i in range(1, 100):
    random_numbers[0] = random_numbers[0] + 2
    first_group.append(random_numbers[0])


first_group.insert(0, first_number)
print(first_group)

first_half = first_group[0:50]
print(first_half)
second_half = first_group[50: 100]
print(second_half)
