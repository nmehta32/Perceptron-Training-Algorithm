# Perceptron-Training-Algorithm
Implementation of Perceptron Training Algorithm from scratch in Python


Here W0, W1, W2 = [-0.009476335089441956, -0.2991156742653023, 0.8507104983388833] which are the perfect 
Weights For classification
Random Weights W0_,W1_,W2_ = [-0.808725,-0.08573708, 0.930979] are selected and fed into the neural network
Number of misclassifications are 49 when using the above random weight
After one epoch
Weights become [[-0.87802575]
[-0.2029027 ]
[-1.14059471]]
And Number of misclassifications reduce to 38
Next,
Epoch Number: 2
Updated weights: [[-0.87802575]
[ 0.88141992]
[-1.73969219]]
Number of missclassifications: 25
And so on until convergence
The final weights with ETA = 1 are [[ 0.12197425] [ 3.36640913] [-8.78704387]] on epoch 47 comparing to the optimal 
weights [-0.009476335089441956, -0.2991156742653023, 0.8507104983388833] they are very different but since the 
perceptron converges they successfully classify the elements
