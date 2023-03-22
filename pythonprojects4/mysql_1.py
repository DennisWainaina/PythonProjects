# Program for checking if a number is in a list and the index of the number in the list
x = [5, 6, 8, 7]
a = 7
count = -1
for i in x:
    count = count + 1
    if i == a:
        print('The number is in the list its index is', count)
        break
else:
    print('The number is not in the list')

n = 20

if n % 2 != 0:
    print('Weird')

if n % 2 == 0 and n in range(2, 5):
    print('Not weird')

if n % 2 == 0 and n in range(6, 21):
    print('Weird')

if n % 2 == 0 and n > 20:
    print('Not weird')


def is_leap(year):
    # Write your logic here
    if year % 4 == 0:
        return True

    if year % 4 != 0:
        return False

    return year


year = int(input())
print(is_leap(year))


def fizzBuzz(n):
    # Write your code here
    for i in range(1, n):
        if i % 3 == 0 and i % 5 == 0:
            print('Fizzbuzz')

        elif i % 3 == 0 and i % 5 != 0:
            print('Fizz')

        elif i % 3 != 0 and i % 5 == 0:
            print('Buzz')

        elif i % 3 != 0 and i % 5 != 0:
            print(i)


if __name__ == '__main__':
    n = int(input().strip())

fizzBuzz(n)


n = str(input())
arr = [int(x) for x in n]
arr.sort(reverse=True)
print(arr)

unique = []

for x in arr:
    if x not in unique:
      unique.append(x)

print(unique)

name_list = []
score_list = []
final_list = []
for i in range(int(input("Please enter the number of students:"))):
    name = input('Please enter your name:')
    score = float(input('Please enter your score:'))
    final = name, score
    other = list(final)
    name_list.append(other)


name_list.sort(reverse=True)
print(name_list)
print(name_list[2][0])
random = [10, 20, 30, 50, 60, 70, 80]
print(random[-1])

a = 1
arr = [-4, 3, -9, 4, 1]
positive_elements = []
negative_elements = []
zero_elements = []
for i in arr:
    if i > 0:
        positive_elements.append(i)
    elif i < 0:
        negative_elements.append(i)
    elif i == 0:
        zero_elements.append(i)

print(len(positive_elements) / len(arr))
print(len(negative_elements) / len(arr))
print(len(zero_elements) / len(arr))
