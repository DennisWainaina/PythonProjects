# Program for calculating income tax

def taxes():
    # The first step is to write the income to be taxed

    income = 200000

    while income > 0:
        # The while loop is to ensure taxing stops when income = 0
        # We then start taxing the money according to the tax brackets

        tax = 0 * 10000

        # We then subtract the first tax bracket from the original income to get new income
        new_income = income - 10000

        if new_income <= 0:
            print("The tax is 0")
            break

        # From the new income a new tax bracket is calculated for the remaining amount
        tax2 = 0.1 * 10000

        # The new tax bracket is then subtracted from the remaining amount
        new_income2 = new_income - 10000

        if new_income2 <= 0:
            print("The tax is ", tax2)
            break

        # The remaining amount is then taxed
        tax3 = 0.2 * new_income2

        # All the taxes are then added together to get the total amount taxed
        tax4 = tax + tax2 + tax3

        # The total tax is then printed
        print("The tax for", income, "is", tax4)

        # Then the amount remaining after tax has been deducted
        remaining = income - tax4
        print("The amount remaining after tax is", remaining)

        # This is to close the while loop
        new_income3 = new_income2 - new_income2
        income = new_income3


taxes()
