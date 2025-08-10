# Summary

`` more like a bunch of keywords to remember what they are ``

* training an algorithm: data, model, objective function, optimization algorithm; preprocessing, one-hot encoding;
* non-linear structures, neural networks, layers, backpropagation, overfitting, underfitting;
* training, validation, testing sets; early stopping;
* batching, learning rate; 
* mnist dataset;

# What's More?

`` Computer Vision, NLP, LLMs... ``

## CNNs
* Convolutional Neural Networks: reduces the dimensiality of the problem by:
    * Convolutional Layer: works like guided filtering works, takes the avg. of 5x5 (nxn) kernels and puts them into one: goes on and on by making it smaller for every iteration.
    * Pooling Layer: does the same thing but without overlapping.
* Used mostly for image recognition and other image-related problems. (visual data)

## RNNs
* Used mostly for sequential data like trading stuff (stocks), music, speech recognition and NLP
* They have memory, since they work with SEQUENTIAL data
    * h1 uses both h0 (hidden layer 0) and x1(input layer 1) to reach y1 (output_layer_1)
* they are computationally expensive

## Non-NN ML Approaches
* Random Forests: uses decision trees to make them overfit less. 
    * makes many bad classifiers equal a good CLASSIFIER
* Generative Models: gives a probability whether an output is correct. the target is joint probability.
    * used for things like translation.