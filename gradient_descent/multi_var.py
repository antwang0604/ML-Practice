import numpy as np
import matplotlib.pyplot as plt

def z_func(x, y):
    return x ** 2+ y**3

def z_gradient(x, y):
    return (2*x, 3*(y**2))

# Points for plotting the function
x = np.arange(-100,100,0.1)
y = np.arange(-100,100,0.1)
z = z_func(x,y)

# Gradient Descent parameters
init_x = 80
init_y = 60
initial_point = (init_x, init_y, z_func(init_x, init_y))

max_iterations = 3000 
min_step_size = 0.0001 
learning_rate = 0.01
step_size_clipping = 50

X,Y = np.meshgrid(x,y)
Z = z_func(X,Y)
ax = plt.subplot(projection="3d", computed_zorder=False)


# initialize the algorithm
current_x = init_x
current_y = init_y

for n in range(max_iterations):
    current_point = (current_x, current_y, z_func(current_x, current_y))
    step_sizes = [z_gradient(current_x, current_y)[0] * learning_rate, z_gradient(current_x, current_y)[1] * learning_rate]
    
    if abs(step_sizes[0]) <= min_step_size or abs(step_sizes[1]) <= min_step_size:
        break

    def sign(x):
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0
    for ind, step_size in enumerate(step_sizes):
        if abs(step_size) > step_size_clipping:
            step_sizes[ind] = sign(step_size) * step_size_clipping

    current_x -= step_sizes[0]
    current_y -= step_sizes[1]

    print(f"iteration: {n} current (x,y): {current_point[0]},{current_point[1]} gradient (x,y): {z_gradient(current_x,current_y)} step_size (x,y): {step_sizes}")
    ax.plot_surface(X,Y,Z,cmap="viridis")
    ax.scatter(initial_point[0],initial_point[1],initial_point[2], color="red", zorder=1)
    ax.scatter(current_point[0],current_point[1],current_point[2], color="purple", zorder=1)
    plt.pause(0.001)
    ax.clear()
plt.show()

