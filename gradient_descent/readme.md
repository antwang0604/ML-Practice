# Gradient Descent From Scratch

## Examples covered:
- EX1: Simple Quadratic Equation: $y=x^2$
- EX2: Exploding Gradient & Oscillation on $y=x^4$
- EX3: Simple Multi Variable Equation $z = x^2 + y^3$

## Concept of Gradient Descent
Gradient Descent works by iteratively adjusting model parameters in the direction that reduces the cost the most, guided by the gradient (slope) of the cost function. This process continues until the algorithm converges to a point where the cost is minimized, effectively finding the best model parameters for a given task. 

In other words, the algorithm will converge when it finds the point where slope equals zero. The effect of gradient descent is when far away from the converging point, the algorithm will take large strides toward the solution, and as the algorithm apporaches solution, it will take baby steps until convergence.

Generalized Gradient Descent formula is
$$
\theta^{t+1} = \theta^t - \gamma \cdot \nabla J(\theta^t) \\[1cm]
\nabla J(\theta_n) = \begin{bmatrix} \frac{\partial J}{\partial\theta_1} \\[0.2cm] \frac{\partial J}{\partial\theta_2} \\ \vdots \\ \frac{\partial J}{\partial\theta_n} \end{bmatrix} \
$$

Here, $J(\theta)$ is the cost function and $\nabla J(\theta)$ is the gradient of the cost function. Starting from an intial point, the gradient descent algorithm will iteratively update the next point. At iteration $t$, the gradient of the cost function at that instance multiplied by the learning rate $\gamma$ equals to step size, which is how far to travel towards the solution. 

## Example 1 - $y=x^2$
In the first example, for a simple single variable cost function $y=x^2$ the derivative is $y=2x$. This means that at each iteration of $x$, if we take the learning rate $\gamma$ multiplied by the slope of the curve at that point, we will get the step size. To get to the next point, we just need to move $x$ by the step size. There are different parameters we can adjust in the process:

### Parameters of Gradient Descent
#### max_iteration: 
the maximum number of iterations algorithm will run before it stops. Setting a maximum number of iterations is a way to prevent the algorithm from running indefinitely, especially when convergence is slow or may not occur due to issues like a poor choice of learning rate.

#### min_step_size: 
AKA tolerance, a stopping criterion for gradient descent. It represents the minimum allowable step size that the algorithm can take. If the step size falls below this threshold, the algorithm is assumed to have converged, and the optimization process is terminated.

#### learning_rate: 
hyperparameter that determines the size of the steps taken in each iteration. It controls how quickly or slowly the algorithm converges to the minimum of the cost function. A larger learning rate leads to faster convergence but may result in overshooting the minimum or even divergence. A smaller learning rate leads to more stable convergence but may require more iterations to reach the minimum.

The result of this example looks like this. 

![]()

We can see that when the alogrithm is far away from the solution, in this case it is when x=0, it moves fast towards the solution. This is because the slope is higher when far away. And as the algorithm closes up towards the solution, the slope decreases and the step sizes become smaller. When the slope gets really small, and step size becomes less than **min_step_size**, the algorithm is considered completed.

## Example 2 - $y=x^4$
There are times when the gradient "explodes" or causes oscillation. Explosion occurs when dealing with a cost function like y=x^4, which has a steep curvature for values of x away from the minimum, it's common for gradient descent to have convergence issues due to large step sizes. This can lead to overshooting the minimum or slow convergence. Below is an example and starting from the first iteration, the slope is alreasy large causing a larget step size, which causes the subsequent iterations to continue to increase.

![](https://github.com/antwang0604/ML-Practice/blob/main/gradient_descent/content/QuadraticGD.gif)

To address this, we introduce a new parameter called gradient clipping.

### Gradient Clipping
Gradient clipping is a technique used to limit the magnitude of the gradient. If the gradient becomes too large, 
it's scaled down to a predefined threshold. This can prevent excessively large steps in regions with steep curvature.

With graident clipping implemented in the code with_clipping.py, we get:

![](https://github.com/antwang0604/ML-Practice/blob/main/gradient_descent/content/ProperClipping.gif)

Oscillation can also occur if the parameters are not properly set:

![](https://github.com/antwang0604/ML-Practice/blob/main/gradient_descent/content/Oscillation.gif)


![](https://github.com/antwang0604/ML-Practice/blob/main/gradient_descent/content/multivar.gif)
