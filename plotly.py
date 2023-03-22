from matplotlib import pyplot as plt
from matplotlib import style

# This is a module used to plot data using python to visualize data on a graph

x = [5, 8, 10]
y = [12, 16, 6]
# These are the coordinates of the graph

plt.plot(x, y)
# This is a command used to plot the graph using the x and y co-ordinates

plt.title('Info')
# This command gives the title of the graph to be plotted

plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

# These commands give the name of the y and x-axis


# This command shows the result of the plot in a visual manner
# You can also add style to graph
# To do this you have to import the style function from matplotlib


style.use('ggplot')
# This gives the style of the graph to be plotted

x = [5, 8, 10]
y = [12, 16, 6]
x2 = [6, 9, 11]
y2 = [6, 15, 7]
# These are the co-ordinates of the graph

plt.plot(x, y, 'g', label='line one', linewidth=5)
plt.plot(x2, y2, 'c', label='line two', linewidth=5)
# g and c are the colors of the two lines plotted by x,y and x2,y2
# label is the name of the line and line-width is the width of the line
# These are used to more details to the graph being drawn


plt.title('Epic info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
# These are the titles of the graph itself the y and x-axis

plt.legend()
# This is used to show the legend on the graph

plt.grid(True, color='k')
# This is used to show gridlines on the graph

plt.show()

n = 27
for i in range(2, n):
    if n % i == 0:
        print(n, 'is not a prime number')
        break
    else:
        print(n, 'is a prime number')
        break
