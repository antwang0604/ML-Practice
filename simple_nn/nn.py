import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


w1_optimal= 3.34
w2_optimal= -3.53
b1_optimal= -1.43
b2_optimal= 0.57

# only for plotting, not for calculations:
w3_optimal = -1.22
w4_optimal = -2.30
b3_optimal = 2.61

w3_init = w3_curr = 0.36
w4_init = w4_curr = 0.63
b3_init = b3_curr = 0

data_set = [[0,0],[0.5,1],[1,0]]

max_iterations = 3000 
learning_rate = 0.1


def HiddenLayerInput (input, weight, bias):
    return ( input * weight ) + bias

def ApplySoftMax (input):
    return np.log(np.exp(input) + 1)

def CalculateOutput (softmax1, weight3, softmax2, weight4, bias):
    return ( softmax1 * weight3 ) + ( softmax2 * weight4 ) + bias

def CalculateHiddenLayerOutput (hidden_node, input):
    if(hidden_node == 1):
        hidden_layer_input_1 = HiddenLayerInput(input,w1_optimal,b1_optimal)
        hidden_layer_output_1 = ApplySoftMax(hidden_layer_input_1)
        return hidden_layer_output_1
    elif(hidden_node == 2):
        hidden_layer_input_2 = HiddenLayerInput(input,w2_optimal,b2_optimal)
        hidden_layer_output_2 = ApplySoftMax(hidden_layer_input_2)
        return hidden_layer_output_2
    else:
        return 0

def CalculatePrediction (input,w3=None,w4=None,b3=None):
    if w3 == None:
        w3 = w3_curr
    if w4 == None:
        w4 = w4_curr
    if b3 == None:
        b3 = b3_curr
    hidden_layer_output_1 = CalculateHiddenLayerOutput(1,input)
    hidden_layer_output_2 = CalculateHiddenLayerOutput(2,input)
    prediction = CalculateOutput(hidden_layer_output_1,w3,hidden_layer_output_2,w4,b3)
    return prediction


def CalculateStepSize_b3 ():
    """
    Predicted = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
    for dJ/db3  = dJ/dPredicted * dPredicted/db3
                = Sum of { -2*(Observed_i - Predicted_i) }
    """
    slope=0
    for dosage, observed in data_set:
        slope += -2 * ( observed - CalculatePrediction(dosage) )
    print(f"b3 slope {slope:.3f}")
    return slope * learning_rate

def CalculateStepSize_w3 ():
    """
    Predicted = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
    for dJ/db3  = dJ/dPredicted * dPredicted/dw3 
                = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_1 }
    """
    slope=0
    for dosage, observed in data_set:
        predicted = CalculatePrediction(dosage)
        slope += -2 * ( observed - CalculatePrediction(dosage)) *  CalculateHiddenLayerOutput(1,dosage)
    print(f"w3 slope {slope:.3f}")
    return slope * learning_rate

def CalculateStepSize_w4 ():
    """
    Predicted = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
    for dJ/db3  = dJ/dPredicted * dPredicted/dw4
                = Sum of { -2*(Observed_i - Predicted_i) * hidden_layer_output_2 }
    """
    slope=0
    for dosage, observed in data_set:
        predicted = CalculatePrediction(dosage)
        slope += -2 * ( observed - CalculatePrediction(dosage)) *  CalculateHiddenLayerOutput(2,dosage)
    print(f"w4 slope {slope:.3f}")
    return slope * learning_rate

x = [point[0] for point in data_set]
y = [point[1] for point in data_set]

def CalculateSSR (w3=w3_init,w4=w4_init,b3=b3_init):
    ssr = 0
    for dosage, observed in data_set:
        predicted = CalculatePrediction(dosage,w3=w3,w4=w4,b3=b3)
        ssr += ( (observed - predicted ) ** 2 )
    return ssr

fig = plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Subplot 1
ax1 = plt.subplot(gs[0])
# Subplot 2
ax2 = plt.subplot(gs[1])
# Subplot 3
ax3 = plt.subplot(gs[2])
# Subplot 4
ax4 = plt.subplot(gs[3])

for n in range(max_iterations):


    b3_step_size = CalculateStepSize_b3()
    w3_step_size = CalculateStepSize_w3()
    w4_step_size = CalculateStepSize_w4()
    b3_curr = b3_curr - b3_step_size
    w3_curr = w3_curr - w3_step_size
    w4_curr = w4_curr - w4_step_size
    print(f"-----------------------------------iteration {n}---------------------------------")
    print(f"b3 step size {b3_step_size:.3f}, curr b3 {b3_curr:.3f}")
    print(f"w3 step size {w3_step_size:.3f}, curr w3 {w3_curr:.3f}")
    print(f"w4 step size {w4_step_size:.3f}, curr w4 {w4_curr:.3f}")
    print(f"dosage {x[0]} - observed {y[0]}, predicted {CalculatePrediction(x[0]):.3f} | dosage {x[1]} - observed {y[1]}, predicted {CalculatePrediction(x[1]):.3f} | dosage {x[2]} - observed {y[2]}, predicted {CalculatePrediction(x[2]):.3f}")

    predict_x = np.arange(0,1.1,0.05)
    predict_y = CalculatePrediction(predict_x)

    ax1.scatter(x, y, label='Data Points')
    ax1.set_xlabel('Dosage')
    ax1.set_ylabel('Efficency')
    ax1.plot(predict_x,predict_y, label='Prediction')

    ssr_x = np.arange(b3_optimal-4,b3_optimal+4,0.5)
    ssr = []
    for value in ssr_x:
        ssr.append(CalculateSSR(w3=w3_optimal,w4=w4_optimal,b3=value))
    ax2.plot(ssr_x, ssr, label='SSR - b3')
    ax2.scatter(b3_curr,CalculateSSR(w3=w3_optimal,w4=w4_optimal,b3=b3_curr),color="red")
    ax2.set_xlabel('b3')
    ax2.set_ylabel('Residual Squared Sum')

    ssr_x = np.arange(w3_optimal-4,w3_optimal+4,0.5)
    ssr = []
    for value in ssr_x:
        ssr.append(CalculateSSR(w3=value,w4=w4_optimal,b3=b3_optimal))
    ax3.plot(ssr_x, ssr, label='SSR - w3')
    ax3.scatter(w3_curr,CalculateSSR(w3=w3_curr,w4=w4_optimal,b3=b3_optimal),color="red")
    ax3.set_xlabel('w3')
    ax3.set_ylabel('Residual Squared Sum')

    ssr_x = np.arange(w4_optimal-4,w4_optimal+4,0.5)
    ssr = []
    for value in ssr_x:
        ssr.append(CalculateSSR(w3=w3_optimal,w4=value,b3=b3_optimal))
    ax4.plot(ssr_x, ssr, label='SSR - w4')
    ax4.scatter(w4_curr,CalculateSSR(w3=w3_optimal,w4=w4_curr,b3=b3_optimal),color="red")
    ax4.set_xlabel('w4')
    ax4.set_ylabel('Residual Squared Sum')

    ax1.set_title(f'Neural Network Prediction & Training Data, iteration {n}')
    ax2.set_title(f'Residual Square Sum vs Param b3, b3={b3_curr:.3f}')
    ax3.set_title(f'Residual Square Sum vs Param w3, w3={w3_curr:.3f}')
    ax4.set_title(f'Residual Square Sum vs Param w4, w4={w4_curr:.3f}')

    plt.pause(0.01)
    plt.tight_layout()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

plt.show()