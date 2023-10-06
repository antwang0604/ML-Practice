import numpy as np
import matplotlib.pyplot as plt

def y_func(x):
    return x ** 2

def y_derivative(x):
    return 2 * (x)

# Points for plotting the function
x = np.arange(-100,100,0.1)
y = y_func(x)

# Gradient Descent parameters
init_x = 80
initial_point = (init_x, y_func(init_x))
max_iterations = 3000 
min_step_size = 0.0001 
learning_rate = 0.01

# initialize the algorithm
current_x = init_x

for n in range(max_iterations):
    current_point = (current_x, y_func(current_x))
    step_size = y_derivative(current_x) * learning_rate

    print(f"iteration: {n} current x: {current_x} slope: {y_derivative(current_x)} step_size: {step_size}")
    if abs(step_size)<=min_step_size:
        break
    current_x -= step_size

    plt.plot(x,y)
    plt.scatter(initial_point[0],initial_point[1],color="red")
    plt.scatter(current_point[0],current_point[1],color="blue")
    plt.pause(0.001)
    plt.clf()


