# DL - TensorFlow 2.0 Intro
# It is a DL library, developed by Google, that allows us to create fairly complicated models with little coding.

# !pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Fake Data Generation
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))
generated_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 2*xs - 3*zs + 5 + noise
# create a .npz file to use as data
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# data -> preprocess -> save in .npz to build your algorithm

# Building the model
training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1
# Dense: takes the inputs, calculates the dot product of the inputs and the weights and adds the bias (in np-> np.dot(inputs,weights)+bias)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

# Optimizer and loss function specifying part
# check tf.keras.optimizers for optimizers you need
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# model.fit(inputs, targets) -> trains the model with given data and the targets
# epoch = iteration over the full dataset
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)
# verbose=0 -> silent, no output about the training is displayed
# verbose=1 -> progress bar
# verbose=2 -> one line per epoch

# Extract the weights and bias

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

print(weights, bias)

# Extract the outputs (make predictions)

# model.predict_on_batch(data) -> calculates the outputs given inputs
model.predict_on_batch(training_data['inputs'])

training_data['targets'].round(1)

# Plot the data (in a successfull result like this, the line should be as close to 45 degrees as possible)

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()