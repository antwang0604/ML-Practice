# Back Propogation of Simple Neural Network

We have a set of data points that represents dosage of medicine vs efficacy of medicine:

![Data Points](https://github.com/antwang0604/ML-Practice/blob/main/simple_nn/content/DataPoints.png)

We want to fit a curve using a simple neural network so we can predict what the efficacy of the medicine is given the dosage. The neural network we can use to model this example has 7 parameters

![Simple Nerual Network](https://github.com/antwang0604/ML-Practice/blob/main/simple_nn/content/SimpleNN.png)

The parameters of the neural network (weights and biases) will determine what the fitted curve looks like and they need to be individually optimized. But assuming that the first four parameters w1, w2, b1, b2 have already been optimized, we want to use back propogation to optimze the remaining three parameteres w3, w4, b3 to demonstrate the concept of back propogation.

In the nerual network, the predicted output can be calculated as (with respect to the unoptimized parameters):

Predicted = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3

where the hidden_layer_output_1 (y1) is the output of the first activation function: log(1+e^x1) where x1 = (w1 * x ) + b1 
and hidden_layer_output_2 (y2) is the output of the second activation function: log(1+e^x2) where x2 = (w2 * x ) + b2
x represents the dosage of medicine.

We can also define our cost function using Residual Squared Sum as:

J = Sum of { (Observed_i - Predicted_i) ** 2}

The goal of the optimization problem is to find a value of w3, w4, and b3 where the cost function will be minimized. We can use gradient descent to solve the optimal values for w3, w4, and b3:

Gradient Descent for each parameter:
b3_next = b3 - (learning_rate * dJ/db3)
w3_next = w3 - (learning_rate * dJ/dw3)
w4_next = w4 - (learning_rate * dJ/dw4)

to calculate the derivative of the cost function with respect to each parameter, we can apply the chain rule and we get:

dJ/db3  = dJ/dPredicted * dPredicted/db3
        = Sum of { -2*(Observed_i - Predicted_i) }

dJ/db3  = dJ/dPredicted * dPredicted/dw3 
        = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_1_i }
        = Sum of { -2*(Observed_i - Predicted_i) * log(1+e^x1) } where x1 = (w1 * dosage ) + b1
    
dJ/db3  = dJ/dPredicted * dPredicted/dw4
        = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_2_i }
        = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_2_i }
        = Sum of { -2*(Observed_i - Predicted_i) * log(1+e^x2) } where x2 = (w2 * dosage ) + b2

We want to take the Sum of {...} meaning that the data points we have will be plugged into the function for the three data points we have. This is "training" the nerual network. 

For example in dJ/db3, Sum of { -2*(Observed_i - Predicted_i) } means that for the three data points [ (0,0), (0.5,1), (1,0) ] we will plug them into the nerual network.

Data point 1 where Observed_1 = 0 and Predicted_1 = NN(0) -> run 0 into the neural network to get the predicted value.
Data point 1 where Observed_2 = 1 and Predicted_1 = NN(0.5) -> run 0.5 into the neural network to get the predicted value.
Data point 1 where Observed_3 = 0 and Predicted_1 = NN(1) -> run 1 into the neural network to get the predicted value.
so Sum of { -2*(Observed_i - Predicted_i) } = -2*(Observed_1 - Predicted_1) -2*(Observed_2 - Predicted_2) -2*(Observed_3 - Predicted_3)

We initialize the three parameters (w3, w4, b3) with a random value, while the remaining parameters are plugged in with the optimal values. In each step, we will calculate the gardient to get the updated value of w3, w4, b3.

The training will continue until the max_iteration has elapsed, or the min_step_size has been reached which results in the final optimal values for w3, w4, and b3. The training process is visualized here:

![Gradient Descent of 3 variables](https://github.com/antwang0604/ML-Practice/blob/main/simple_nn/content/NN_GD.gif)

Top left shows the fitted curve - how well the neural network is able to fit the data points. Overtime, the fitted curve will become 
Top right shows the Cost Function J with different values of w3. Note, w4 and b3 are already plugged in with optimal values.
Bottom left shows the Cost Function J with different values of w4. Note, w3 and b3 are already plugged in with optimal values.
Bottom right show the Cost Function J with different values of b3. Note, w3 and w4 are already plugged in with optimal values.

