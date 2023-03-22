class Item:
    # The init method is used for defining the instance variables to be used
    def __init__(self, name, age, weight, height, length):
        self.name = name
        self.age = age
        self.weight = weight
        self.height = height
        self.length = length

    def new_age(self):
        return self.age + 2

    def names(self):
        print(self.name)

    def weights(self, extra):
        print(self.weight)
        extra = self.weight + ' heavier'
        print(extra)

    @staticmethod
    def height(height):
        new_height = height * 2.54
        return new_height

    @property
    def long(self):
        return self.length


book = Item("Dennis", 6, "6 grammes", 72, 56)
print(book.new_age())
book.names()
book.weights('you')
print('The height in centimeters is', Item.height(72))
print(book.long)
