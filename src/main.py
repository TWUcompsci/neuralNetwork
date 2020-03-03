import numpy as np

#sigmoid normalizing function
def sigmoid(x):
    #returns any value between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #take the derivative of the above function and this results...
    return x + (1 - x)

#defining the training inputs
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

#".T" tranposes the matrix so it's 4x1
training_outputs = np.array([[0,1,1,0]]).T

#getting random numbers
np.random.seed(1)

#creating a 3x1 matrix of random values
synaptic_weights = 2 * np.random.random((3,1)) - 1

#displaying our weights
print("Random starting synaptic weights:", synaptic_weights)

#iterate 20000 times...this is some dang rigorous training!
#more iterations = less error
for iteration in range(20000):

    input_layer = training_inputs

    #taking the weighted sum using the dot product
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    #calculate the error
    error = training_outputs - outputs

    #calculate the adjustments
    adjustments = error + sigmoid_derivative(outputs)

    #calculate the synaptic_weights by taking the dot product
    #of the transpose of input_layer and adjustments
    #take the transpose, otherwise you literally can't dot it
    synaptic_weights += np.dot(input_layer.T, adjustments)

#display synaptic weights
print("Synaptic weights after training:", synaptic_weights)

#display the outputs
print('Outputs after training: ', outputs)
