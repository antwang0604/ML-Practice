# Gradient Descent From Scratch

Examples:
- Simple Quadratic Equation: $y=x^2$
- Exploding Gradient & Oscillation on $y=x^4$

## Concept of Gradient Descent
Gradient Descent works by iteratively adjusting model parameters in the direction that reduces the cost the most, guided by the gradient (slope) of the cost function. This process continues until the algorithm converges to a point where the cost is minimized, effectively finding the best model parameters for a given task.

Generalized Gradient Descent formula
$$
\theta^{t+1} = \theta^t - \gamma \cdot \nabla J(\theta^t) \\[1cm]
\nabla J(\theta_n) = \begin{bmatrix} \frac{\partial J}{\partial\theta_1} \\[0.2cm] \frac{\partial J}{\partial\theta_2} \\ \vdots \\ \frac{\partial J}{\partial\theta_n} \end{bmatrix} \
$$

## Parameters
### max_iteration: 
the maximum number of iterations algorithm will run before it stops. Setting a maximum number of iterations is a way to prevent the algorithm from running indefinitely, especially when convergence is slow or may not occur due to issues like a poor choice of learning rate.

### min_step_size: 
AKA tolerance, a stopping criterion for gradient descent. It represents the minimum allowable step size that the algorithm can take. If the step size falls below this threshold, the algorithm is assumed to have converged, and the optimization process is terminated.

### learning_rate: 
hyperparameter that determines the size of the steps taken in each iteration. It controls how quickly or slowly the algorithm converges to the minimum of the cost function. A larger learning rate leads to faster convergence but may result in overshooting the minimum or even divergence. A smaller learning rate leads to more stable convergence but may require more iterations to reach the minimum.

## Gradient Clipping
When dealing with a cost function like y=x^4, which has a steep curvature for values of x away from the minimum, 
it's common for gradient descent to have convergence issues due to large step sizes. This can lead to overshooting the minimum or slow convergence.
To address this: Gradient Clipping: Gradient clipping is a technique used to limit the magnitude of the gradient. If the gradient becomes too large, 
it's scaled down to a predefined threshold. This can prevent excessively large steps in regions with steep curvature.