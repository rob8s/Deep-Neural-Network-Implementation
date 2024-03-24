#Robert Bates
#02/21/2024
#Program 1
#DNN implementation

#imports
import numpy as np
import argparse
import sys

#activations funcs and their derivatives
def relu(matrix):
    return np.maximum(0, matrix) 

def relu_der(matrix):
    return np.where(matrix>0, matrix, 0)

def sigmoid(matrix):
    return 1/(1+np.exp(-matrix))

def sigmoid_der(matrix):
    return sigmoid(matrix) * (1-sigmoid(matrix))

def tanh(matrix):
    return np.tanh(matrix)

def tanh_der(matrix):
    return 1 - np.tanh(matrix)**2    

def identity(matrix):
    return matrix

def identity_der(matrix):
    return matrix

def softmax(matrix):
    softed = np.exp(matrix - np.max(matrix))
    return softed / softed.sum()


#argparse command requirements
parser = argparse.ArgumentParser()
#verbose mode
parser.add_argument('-v')
#all data, train and dev
parser.add_argument('-train_feat', required=True)
parser.add_argument('-train_target', required=True)
parser.add_argument('-dev_feat', required=True)
parser.add_argument('-dev_target', required=True)
#epochs 
parser.add_argument('-epochs', required=True, type=int)
#step
parser.add_argument('-learnrate', required=True, type=np.float64)
#hidden units size
parser.add_argument('-nunits', required=True, type=int)
#reg or class
parser.add_argument('-type', required=True)
#hidden activation function
parser.add_argument('-hidden_act', required=True)
#random generated numbers for bias and weights
parser.add_argument('-init_range', required=True, type = float)
#output size
parser.add_argument('-num_classes')
#mb size
parser.add_argument('-mb')
#number of hidden layers
parser.add_argument('-nlayers')

args = parser.parse_args()

#each data piece as a matrix
train_feat_data = np.loadtxt(args.train_feat, dtype=np.float64)
train_target_data = np.loadtxt(args.train_target, dtype=np.float64)
dev_feat_data = np.loadtxt(args.dev_feat, dtype=np.float64)
dev_target_data = np.loadtxt(args.dev_target, dtype=np.float64)

#onehot for target data
if(args.type == 'C'):
    #int form for interpetation
    train_target_data = np.loadtxt(args.train_target, dtype=int)
    dev_target_data = np.loadtxt(args.dev_target, dtype=int)
    #training target data 
    hot_feat_target = np.zeros((train_target_data.size, train_target_data.max() + 1), dtype=np.float64)
    hot_feat_target[np.arange(train_target_data.size), train_target_data] = 1
    train_target_data = hot_feat_target
    
    #Dev target data
    hot_dev_target = np.zeros((dev_target_data.size, dev_target_data.max() + 1), dtype=int)
    hot_dev_target[np.arange(dev_target_data.size), dev_target_data] = 1
    dev_target_data = hot_dev_target

#Normalization of Data for Regression
if(args.type == 'R'):
    train_feat_data = (train_feat_data - np.mean(train_feat_data)) / np.std(train_feat_data)
    train_target_data = (train_target_data - np.mean(train_target_data)) / np.std(train_target_data)
    dev_feat_data = (dev_feat_data - np.mean(dev_feat_data)) / np.std(dev_feat_data)
    dev_target_data = (dev_target_data - np.mean(dev_target_data)) / np.std(dev_target_data)


#overaching biases and weights
current_weights = []
current_bias = []

#generation of random weight matrices, takes into accound size of NN
def generate_weight(hidden_layer_size, input_size, init_range, output_size, total):
    global current_weights
    #zero hidden layers
    if(total == 1):
        current_weights.append(np.random.uniform(-init_range, init_range, size=(input_size, output_size)))
    #with hidden layers
    else:
        #first layer, takes into account the input size
        current_weights.append(np.random.uniform(-init_range, init_range, size=(input_size, hidden_layer_size)))
        #all other layers, with hidden layer dimensions
        for i in range(total-2):
            current_weights.append(np.random.uniform(-init_range, init_range, size=(hidden_layer_size, hidden_layer_size)))
        #final layer, for output size
        current_weights.append(np.random.uniform(-init_range, init_range, size=(hidden_layer_size, output_size)))

#generation of random bias matrices, takes into accound size of NN
def generate_bias(hidden_layer_size, init_range, output_size, total):
    global current_bias
    #zero hidden layers
    if(total == 1):
        current_bias.append(np.random.uniform(-init_range, init_range, size=(1, output_size)))
    #with hidden layers
    else:
        #all layers except for last, hidden layer size
        for i in range(total-1):
            current_bias.append(np.random.uniform(-init_range, init_range, size=(1, hidden_layer_size)))
        #takes into account output size
        current_bias.append(np.random.uniform(-init_range, init_range, size=(1, output_size)))

#getting required number of layers
#output layer only
if(args.nlayers == None):
    add_layer = 1
#hidden layers
else:
    add_layer = int(args.nlayers) + 1

#setting output dimension
if(args.num_classes == None):
    output_size = 1 
else:
    output_size = int(args.num_classes)

#mb to int
if(args.mb == None):
    mb = 1
else:
    mb = int(args.mb)
    if(mb == 0):
        mb = len(train_feat_data)

#activations
activations = []
activations_derivatives = []

#output for each layer, and new gradients
current_outputs = []
gradients = []

#filling activations
if(args.hidden_act == 'sig'):
    for i in range(add_layer-1):
        activations.append(sigmoid)
        activations_derivatives.append(sigmoid_der)

elif(args.hidden_act == 'tanh'):
    for i in range(add_layer-1):
        activations.append(tanh)
        activations_derivatives.append(tanh_der)

elif(args.hidden_act == 'relu'):
    for i in range(add_layer-1):
        activations.append(relu)
        activations_derivatives.append(relu_der)

#regression
if(args.type == 'R'):
    activations.append(identity)
    activations_derivatives.append(identity)

#classification
if(args.type == 'C'):

    #multi
    if(output_size>1):
        activations.append(softmax)
        activations_derivatives.append(identity)

    #simple
    if(output_size==1):
        activations.append(sigmoid)
        activations_derivatives.append(identity)

#train
def train(input):
    global current_outputs
    global current_weights

    #input of overall network, needed for backprop
    current_outputs.append(input)
    #for all layers
    for i in range(len(current_weights)):
        #performs operation at at given layer
        output_i = activations[i](((input @ current_weights[i]) + current_bias[i]))
        #adds to outputs
        current_outputs.append(output_i)
        #updates input for next go around
        input = output_i

#loss function
def loss(target):
    global current_outputs

    #final output-target values
    loss = current_outputs[-1] - target.reshape(current_outputs[-1].shape)
    #updates last output layer to the loss, to initiate gradient calculations
    del current_outputs[-1]
    current_outputs.append(loss)
    #need loss for output layer bias
    gradients.append(loss)
 
#get required gradients for weight and bias updates
def backprop():
    global gradients
    global current_weights
    global current_outputs
    global activations

    #reversed weights and activation derivatives, cost of "list layers"
    rev_weights = current_weights[::-1]
    rev_act = activations_derivatives[::-1]

    for i in range(len(current_weights)-1):
        #the gradient from the previous coming in mult against the weight
        grad = (gradients[i] @ rev_weights[i].T)
        #activations derivative, skips over output derivative
        activated = rev_act[i+1](grad)
        #add to gradients 
        gradients.append(activated)

#update weights and biases
def update_gradients(step):
    global gradients
    global current_weights
    global current_bias
    global current_outputs

    #reverse gradients list, easier to work with
    grad_rev = gradients[::-1]

    for i in range(len(current_weights)):
        #generates new weight
        new_weight = (current_outputs[i].T @ grad_rev[i]) / mb
        #gets new weight
        current_weights[i] = current_weights[i] - (step * new_weight)
        #updates bias
        current_bias[i] = np.sum(grad_rev[i], axis=0) / mb


#takes input and puts it through the current layer setup, returns the output
def run(input):
    global current_weights
    global current_bias
    global activations

    #runs for every layer
    for i in range(len(current_weights)):
        output_i = activations[i]((input @ current_weights[i]) + current_bias[i])
        input = output_i
    return input

#Classification
def accuracy(input, output):
    #converting probabilities to binary
    classes = np.zeros_like(input)
    classes[np.arange(len(input)), np.argmax(input, axis=1)] = 1
    #checking for number of matches to target
    correct = np.sum(classes == output)
    return correct / input.size

#loss funciton for regression
def reg_loss(input, output):
    return np.mean((input - output)**2)

#print out
updates = 1
epochs = 1
#optimal weights and biases
best_weights = []
best_biases = []
#best loss
best_loss = 0
#displaying evaluation data for training and dev sets when specified
#parameters only applicable for verbose mode, the minibatch being currently trained on
def performance(mb_train, mb_target):
    global updates
    global epochs
    global best_weights
    global best_biases
    global best_loss
    
    #loss function for given type
    if(args.type == 'R'):
        loss = reg_loss
    else:
        #classification
        loss = accuracy

    #Train data eval
    train_output = run(mb_train)
    train_tot = loss(train_output, mb_target)

    #dev set eval
    dev_output = run(dev_feat_data)
    dev_tot = loss(dev_output, dev_target_data)

    #verbose mode
    if(args.v != None):
        padded_number = '{:06d}'.format(updates)
        print('Update ' + padded_number + ': train=' + str(round(train_tot, 3)) + ' dev=' + str(round(dev_tot, 3)), file=sys.stderr)
        updates += 1
    #none verbose mode
    else:
        padded_number = '{:03d}'.format(epochs)
        print('Epoch ' + padded_number + ': train=' + str(round(train_tot, 3)) + ' dev=' + str(round(dev_tot, 3)), file=sys.stderr)
        epochs += 1

    #Storing best loss, weights, and biases for potential use
    #regression and the loss is lower, "better"
    if(args.type == 'R' and dev_tot <= best_loss):
        best_loss = dev_tot
        best_weights = current_outputs
        best_biases = current_bias

    #classification, when the accuracy is higher, "better"
    if(args.type == 'C' and dev_tot >= best_loss):
        best_loss = dev_tot
        best_weights = current_outputs
        best_biases = current_bias  

#Training start
        
#generate given weight vectors
generate_weight(args.nunits, len(train_feat_data[0]), args.init_range, output_size, add_layer)
#generate biases
generate_bias(args.nunits, args.init_range, output_size, add_layer)

#setup for mb sizes, and incrementing through overall matrix
if(mb == 0):
    increase = len(train_feat_data)
    run_length = 1
else:
    increase = mb
    run_length = len(train_feat_data) // increase

#overall trianing loop
for i in range(args.epochs):
    #shuffled data
    new_order = np.random.permutation(len(train_target_data))
    feat = train_feat_data[new_order] 
    target = train_target_data[new_order]

    #position of mini batch
    a = 0
    b = increase
    #for # of mb in a given epoch
    for x in range(run_length):
        #reset for next pass
        gradients = []
        current_outputs = []
        #mb's
        train_mb = feat[a:b]
        target_mb = target[a:b]
        #training
        train(train_mb)
        loss(target_mb)
        backprop()
        update_gradients(args.learnrate)
        #next mb
        a += increase
        b += increase

        #verbose mode
        if(args.v != None):
            performance(train_mb, target_mb)
    
    #non-verbose mode
    if(args.v == None):
        performance(train_feat_data, train_target_data)

    

    
