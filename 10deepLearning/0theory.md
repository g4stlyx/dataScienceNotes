# Deep Learning and Deep Neural Networks

* some concepts covered in this part: tensorflow, mnist, customer acquisition
* how to stack and activate layers
* underfitting and overfitting
* training, validation, fold-cross validation, testing, early stopping
* initialization
* optimizers
* preprocessing, normalization, standardization, one-hot encoding

## basics
* 3 major types of machine learning: supervised, unsupervised, reinforcement
* data, model, objective function, optimization algorithm

### linear model
``y = xw + b`` -> ``y= x1w1 + x2w2 + b``
* no matter the number of inputs and outputs, the formula is the same snice all these variables (x,w,b) are vectors, not scalars. 
    * you may need visualization for this one, but you can think these variables as matrices, not simple numbers.

## tensorflow 2.0
* tensor: an algebraic object that describes a multilinear relationship between sets of algebraic objects associated with a vector space.
    * scalars = rank 0 tensors
    * vectors = rank 1 tensors
    * matrices = rank 2 tensors
* scikit_learn for preprocessing, clustering etc.
* tensorflow for DL

## Deep NNs (Deep Nets), Layers 
* layer = the building block of NNs.
    * 1+ layered NN = deep NN = deep net
    * layer types:
        * input layer: data we have
        * output layer: what we compare to our targets
        * hidden layer: we dont know whats happening inside, its name comes from that.
* parameters: found by optimizing
    * weights and biases
* hyper parameters: preset by us
    * width (# hidden units), depth (# hidden layers), learning rate

### Activation Func.s
* transforms inputs into outputs of a different kind. (non-linearities)
    * sigmoid (logistic func.) -> (0,1)
    * ReLu (rectified linear unit) -> (0, infinity)
    * TanH (hyperbolic tangent) -> (-1,1)
    * softmax -> (0,1)

### Backpropogation
* returning from outputs through inputs to correct errors

### overfitting and underfitting
* overfitting: our training has focused on the particular training set so much, it has "missed the point"
* underfitting: the model hasnt captured the underlying logic of the data

### validation
* helps us prevent overfitting using validation dataset.
    * gradient-descent: with the each epoch (iteration), training loss should decrease.
        * validation loss should not increase either, if it does: we are overfitting
* initialization: ps. we set the initial values of weights

### preprocessing
* any manipulation on dataset before running it through the model
    * changing data extension, 
    * giving weights(orders of magnitude) = standardization, feature scaling,
    * relative changes (e.g stock prices increased by 10% COMPARED TO yesterday)