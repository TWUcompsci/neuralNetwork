import numpy as np

def signoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([
                           [0,0,1]
                           [1,1,1]
                           [1,0,1]
                           [0,1,1]
                                 ])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random((3,1))-1

print('random starting weights: ')
print(synaptic_weights)
