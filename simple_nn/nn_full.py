import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# only for plotting, not for calculations:
w1_optimal= 3.34
w2_optimal= -3.53
b1_optimal= -1.43
b2_optimal= 0.57
w3_optimal = -1.22
w4_optimal = -2.30
b3_optimal = 2.61


w1_init = w1_curr = 2
w2_init = w2_curr = -2
b1_init = b1_curr = 0
b2_init = b2_curr = 0
w3_init = w3_curr = -2
w4_init = w4_curr = -3
b3_init = b3_curr = 0

data_set = [[0,0],[0.5,1],[1,0]]

max_iterations = 3000 
learning_rate = 0.05
min_step_size = 0.001


def HiddenLayerInput (input, weight, bias):
    return ( input * weight ) + bias

def ApplySoftMax (input):
    return np.log(np.exp(input) + 1)

def CalculateOutput (softmax1, weight3, softmax2, weight4, bias):
    return ( softmax1 * weight3 ) + ( softmax2 * weight4 ) + bias

def CalculateHiddenLayerOutput (hidden_node, input, weight, bias):
    if(hidden_node == 1):
        hidden_layer_input_1 = HiddenLayerInput(input,weight,bias)
        hidden_layer_output_1 = ApplySoftMax(hidden_layer_input_1)
        return hidden_layer_output_1
    elif(hidden_node == 2):
        hidden_layer_input_2 = HiddenLayerInput(input,weight,bias)
        hidden_layer_output_2 = ApplySoftMax(hidden_layer_input_2)
        return hidden_layer_output_2
    else:
        return 0

def CalculatePrediction (input,w3=None,w4=None,b3=None,w1=None,w2=None,b1=None,b2=None):
    if w1 == None:
        w1 = w1_curr
    if w2 == None:
        w2 = w2_curr
    if b1 == None:
        b1 = b1_curr
    if b2 == None:
        b2 = b2_curr
    if w3 == None:
        w3 = w3_curr
    if w4 == None:
        w4 = w4_curr
    if b3 == None:
        b3 = b3_curr
    hidden_layer_output_1 = CalculateHiddenLayerOutput(1,input,w1,b1)
    hidden_layer_output_2 = CalculateHiddenLayerOutput(2,input,w2,b2)
    prediction = CalculateOutput(hidden_layer_output_1,w3,hidden_layer_output_2,w4,b3)
    return prediction


def CalculateStepSize_w1 ():
    """
    Predicted   = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
                = ( CalculateSoftMax(x1) * w3 ) + ( CalculateSoftMax(x2) * w4 ) + b3
                = ( CalculateSoftMax(input * w1 + b1) * w3 ) + ( CalculateSoftMax(input * w2 + b2) * w4 ) + b3

    for dJ/dw1  = dJ/dPredicted * dPredicted/dy1 * dy1/dx1 * dx1/dw1
                = Sum of { -2*(Observed_i - Predicted_i) * w3 * (e^x1 / (1+e^x1) ) * input }
    """
    slope=0
    for dosage, observed in data_set:
        x1 = HiddenLayerInput(dosage, w1_curr, b1_curr)
        slope += -2 * ( observed - CalculatePrediction(dosage) ) * w3_curr * (math.exp(x1) / (1+math.exp(x1))) * dosage
    print(f"w1 slope {slope:.3f}")
    return slope * learning_rate

def CalculateStepSize_w2 ():
    """
    Predicted   = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
                = ( CalculateSoftMax(x1) * w3 ) + ( CalculateSoftMax(x2) * w4 ) + b3
                = ( CalculateSoftMax(input * w1 + b1) * w3 ) + ( CalculateSoftMax(input * w2 + b2) * w4 ) + b3

    for dJ/dw2  = dJ/dPredicted * dPredicted/dy2 * dy2/dx2 * dx2/dw2
                = Sum of { -2*(Observed_i - Predicted_i) * w4 * (e^x2 / (1+e^x2) ) * input }
    """
    slope=0
    for dosage, observed in data_set:
        x2 = HiddenLayerInput(dosage, w2_curr, b2_curr)
        slope += -2 * ( observed - CalculatePrediction(dosage) ) * w4_curr * (math.exp(x2) / (1+math.exp(x2))) * dosage
    print(f"w2 slope {slope:.3f}")
    return slope * learning_rate

def CalculateStepSize_b1 ():
    """
    Predicted   = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
                = ( CalculateSoftMax(x1) * w3 ) + ( CalculateSoftMax(x2) * w4 ) + b3
                = ( CalculateSoftMax(input * w1 + b1) * w3 ) + ( CalculateSoftMax(input * w2 + b2) * w4 ) + b3

    for dJ/db1  = dJ/dPredicted * dPredicted/dy1 * dy1/dx1 * dx1/db1
                = Sum of { -2*(Observed_i - Predicted_i) * w3 * (e^x1 / (1+e^x1) ) }
    """
    slope=0
    for dosage, observed in data_set:
        x1 = HiddenLayerInput(dosage, w1_curr, b1_curr)
        slope += -2 * ( observed - CalculatePrediction(dosage) ) * w3_curr * (math.exp(x1) / (1+math.exp(x1)))
    print(f"b1 slope {slope:.3f}")
    return slope * learning_rate

def CalculateStepSize_b2 ():
    """
    Predicted   = ( hidden_layer_output_1 * w3 ) + ( hidden_layer_output_2 * w4 ) + b3
                = ( CalculateSoftMax(x1) * w3 ) + ( CalculateSoftMax(x2) * w4 ) + b3
                = ( CalculateSoftMax(input * w1 + b1) * w3 ) + ( CalculateSoftMax(input * w2 + b2) * w4 ) + b3

    for dJ/db2  = dJ/dPredicted * dPredicted/dy2 * dy2/dx2 * dx2/db2
                = Sum of { -2*(Observed_i - Predicted_i) * w4 * (e^x2 / (1+e^x2) ) }
    """
    slope=0
    for dosage, observed in data_set:
        x2 = HiddenLayerInput(dosage, w2_curr, b2_curr)
        slope += -2 * ( observed - CalculatePrediction(dosage) ) * w4_curr * (math.exp(x2) / (1+math.exp(x2)))
    print(f"b2 slope {slope:.3f}")
    return slope * learning_rate


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
        slope += -2 * ( observed - CalculatePrediction(dosage)) *  CalculateHiddenLayerOutput(1,dosage,w1_curr,b1_curr)
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
        slope += -2 * ( observed - CalculatePrediction(dosage)) *  CalculateHiddenLayerOutput(2,dosage,w2_curr,b2_curr)
    print(f"w4 slope {slope:.3f}")
    return slope * learning_rate

def CalculateSSR ():
    ssr = 0
    for dosage, observed in data_set:
        predicted = CalculatePrediction(dosage)
        ssr += ( (observed - predicted ) ** 2 )
    return ssr

x = [point[0] for point in data_set]
y = [point[1] for point in data_set]

for n in range(max_iterations):
    w1_step_size = CalculateStepSize_w1()
    w2_step_size = CalculateStepSize_w2()
    b1_step_size = CalculateStepSize_b1()
    b2_step_size = CalculateStepSize_b2()
    b3_step_size = CalculateStepSize_b3()
    w3_step_size = CalculateStepSize_w3()
    w4_step_size = CalculateStepSize_w4()

    if(abs(w1_step_size) < min_step_size and abs(w2_step_size) < min_step_size and abs(w3_step_size) < min_step_size and abs(w4_step_size) < min_step_size and abs(b1_step_size) < min_step_size and abs(b2_step_size) < min_step_size and abs(b3_step_size) < min_step_size) :
        break

    w1_curr = w1_curr - w1_step_size
    w2_curr = w2_curr - w2_step_size
    b1_curr = b1_curr - b1_step_size
    b2_curr = b2_curr - b2_step_size
    b3_curr = b3_curr - b3_step_size
    w3_curr = w3_curr - w3_step_size
    w4_curr = w4_curr - w4_step_size

    print(f"-----------------------------------iteration {n}---------------------------------")
    print(f"w1 step size {w1_step_size:.3f}, curr w1 {w1_curr:.3f}")
    print(f"w2 step size {w2_step_size:.3f}, curr w2 {w2_curr:.3f}")
    print(f"b1 step size {b1_step_size:.3f}, curr b1 {b1_curr:.3f}")
    print(f"b2 step size {b2_step_size:.3f}, curr b2 {b2_curr:.3f}")
    print(f"w3 step size {w3_step_size:.3f}, curr w3 {w3_curr:.3f}")
    print(f"w4 step size {w4_step_size:.3f}, curr w4 {w4_curr:.3f}")
    print(f"b3 step size {b3_step_size:.3f}, curr b3 {b3_curr:.3f}")
    print(f"dosage {x[0]} - observed {y[0]}, predicted {CalculatePrediction(x[0]):.3f} | dosage {x[1]} - observed {y[1]}, predicted {CalculatePrediction(x[1]):.3f} | dosage {x[2]} - observed {y[2]}, predicted {CalculatePrediction(x[2]):.3f}")

    predict_x = np.arange(0,1.05,0.05)
    predict_y = CalculatePrediction(predict_x)
    textstr = '\n'.join((
    r'$w1=%.2f$' % (w1_curr, ),
    r'$w2=%.2f$' % (w2_curr, ),
    r'$w3=%.2f$' % (w3_curr, ),
    r'$w4=%.2f$' % (w4_curr, ),
    r'$b1=%.2f$' % (b1_curr, ),
    r'$b2=%.2f$' % (b2_curr, ),
    r'$b3=%.2f$' % (b3_curr, ),
    r'$Residual Sq Sum=%.2f$' % (CalculateSSR(), ),
    ))
    plt.title(f'Neural Network Prediction & Training Data, iteration {n}')
    plt.text(0.05, 0.95, textstr,fontsize=8, verticalalignment='top')
    plt.scatter(x, y, label='Data Points')
    plt.xlabel('Dosage')
    plt.ylabel('Efficency')
    plt.plot(predict_x,predict_y, label='Prediction')
    plt.pause(0.002)
    plt.clf()


