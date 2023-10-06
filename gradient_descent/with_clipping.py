import numpy as np
import matplotlib.pyplot as plt

def y_func(x):
    return x ** 4

def y_derivative(x):
    return 4 * (x ** 3)

# Points for plotting the function
x = np.arange(-100,100,0.1)
y = y_func(x)

# Gradient Descent parameters
init_x = 60
initial_point = (init_x, y_func(init_x))
max_iterations = 3000 
min_step_size = 0.0001 
learning_rate = 0.001
step_size_clipping = 100

# initialize the algorithm
current_x = init_x

for n in range(max_iterations):
    current_point = (current_x, y_func(current_x))
    step_size = y_derivative(current_x) * learning_rate
    if abs(step_size) > step_size_clipping:
        if step_size > 0:
            step_size = step_size_clipping
        else:
            step_size = -step_size_clipping

    print(f"iteration: {n} current x: {current_x} slope: {y_derivative(current_x)} step_size: {step_size}")
    if abs(step_size)<=min_step_size:
        break
    current_x -= step_size

    plt.plot(x,y)
    plt.scatter(initial_point[0],initial_point[1],color="red")
    plt.scatter(current_point[0],current_point[1],color="blue")
    plt.pause(0.001)
    plt.clf()


