# This is a module used to make a choice between random data
import random
import random as rand
names = ['Ted', 'Dennis', 'Duro']

# The choice option chooses an option from a list of data
print(rand.choice(names))

# The random option chooses a random value from 0 to less than 1 but never 1
value = rand.random()
print(value)

# We can also get random floating values between two numbers using the uniform option for example
decimal = rand.uniform(1, 10)
print(decimal)

# We can also get random integer values between two numbers using the randint option for example
dice = rand.randint(1, 6)
dice1 = rand.randint(1, 6)
print(dice, dice1)
# This is used to get random values of a die toss between two dies

# There is also the choices option which given a list of data gives a number of times a random choice from the data
# This is shown in the example below

colors = ['Red', 'Black', 'Green']
print(random.choices(colors, k=10))

# We can also change the probability of a certain choice appearing by using the weights option for example
print(random.choices(colors, weights=[18, 18, 2], k=10))
# By doing this we have altered the chance of red and black by 18 out of the total which is 38 and green by 2 out of 38

# From a list of values we can also shuffle the list in random order for example in a deck of cards as shown below
deck = list(range(1, 100))
print(deck)
# The first step is to print the list of cards from 1 to 52

# We then shuffle the list randomly as shown below
random.shuffle(deck)
print(deck)

# We can also obtain a random sample of a specific number for example 5 from a list of data like the one shown above
# For example;
print(random.sample(deck, k=5))

# We can use all these options in the random value to select random data from a group of given data
