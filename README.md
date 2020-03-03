# Simple Neural Network
---
Simple neural network using python

### Follow this tutorial!
<https://www.youtube.com/watch?v=kft1AJ9WVDk>

##### Training Process
1. Take the inputs from the training example and put them through our formula to get the neuron's output
2. Calculate the error, which is the difference between the output we got and the actual output
3. Depending on the severeness of the error, adjust the weights accordingly
4. Repeat the process 20,000 times!

!! Sigmoid Normalizing Function
!! phi(x) = 1 / (1 + e^(-x))

! Adjusting Weights (Error Weight Derivative)
! Use error.input.phi'(output)
! error = output - actual output
! input = 1 or 0
! phi'(x) = x
