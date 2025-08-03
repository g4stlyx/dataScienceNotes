# it is the "hello world" of machine learning. contains 70k grayscale (28x28) images of handwritten digits (0-9=10 classes) and a test set of 10k images.
"""
plan: 
    1. prepare our data and preprocess it: creating training, validation, and test sets.
    2. outline the model and choose the activation functions.
    3. set the appropriate advanced optimizers and the loss function.
    4. make it learn
    5. test the accuracy of the model.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#! 1. Data and Preprocessing
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True , as_supervised=True) # as_supervised=True, loads the dataset as a tuple (image, label)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples # we will reserve ~6k samples for validation
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples # 10k samples for testing
num_test_samples = tf.cast(num_test_samples, tf.int64)

def scale(image, label):
    image = tf.cast(image, tf.float32) # convert to float32
    image /= 255.0 # scale to [0, 1] from [0, 255]
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale) # map the scale function to the training set
scaled_test_data = mnist_test.map(scale) # map the scale function to the test set

# shuffle the training data
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples) # take the first 6k samples for validation
train_data = shuffled_train_and_validation_data.skip(num_validation_samples) # skip the first 6k samples for training

BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE) # batch the training data
validation_data = validation_data.batch(num_validation_samples) # batch the validation data
test_data = scaled_test_data.batch(num_test_samples) # batch the test data

validation_inputs, validation_targets = next(iter(validation_data)) # get the validation inputs and targets

#! 2. Outline the Model
# 28x28 = 784 input nodes, 10 output nodes (one for each digit-class-)
input_size = 784
output_size = 10
hidden_layer_size = 100 # number of hidden nodes

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # flatten the input images to a vector of size 784
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # hidden layer with ReLU activation
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # hidden layer with ReLU activation
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer with softmax
])

#! 3. Set the Optimizer and Loss Function
model.compile(optimizer='adam', # advanced optimizer
              loss='sparse_categorical_crossentropy', # loss function for multi-class classification
              metrics=['accuracy']) # metrics to track during training

#! 4. Make it Learn
NUM_EPOCHS = 5 # number of epochs to train the model
model.fit(train_data, 
          epochs=NUM_EPOCHS, 
          validation_data=(validation_inputs, validation_targets), 
          verbose=2) # verbose=2 for detailed output during training

#! 5. Test the Accuracy of the Model
test_loss, test_accuracy = model.evaluate(test_data, verbose=2) # evaluate the model on the test data
print('Test loss: {:.2f}. Test Accuracy: {:.2f}%'.format(test_loss, test_accuracy * 100.))