# Back Propogation of Simple Neural Network

The neural network as 7 parameters, but assuming that the first four parameters w1, w2, b1, b2 have already been optimized, we want to use back propogation to optimze the remaining three parameteres w3, w4, b3.

Predicted = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
Residual Squared Sum (Cost Function) = J = Sum of { (Observed_i - Predicted_i) ** 2}

dJ/db3  = dJ/dPredicted * dPredicted/db3
        = Sum of { -2*(Observed_i - Predicted_i) }

dJ/db3  = dJ/dPredicted * dPredicted/dw3 
        = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_1 }
    
dJ/db3  = dJ/dPredicted * dPredicted/dw4
        = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_2 }

Gradient Descent:
b3_next = b3 - (learning_rate * dJ/db3)
w3_next = w3 - (learning_rate * dJ/dw3)
w4_next = w4 - (learning_rate * dJ/dw4)

