""" task:
! predicting if a customer buys another book from a book store, based on their past purchase behavior and book reviews etc.
2 years of data to train the model. 6 months of data after that 2 years to create targets (1 if they buy another book, 0 if not): ground truth
"""

""" preprocessing example:
it is bad to have a reviews column with too many missing values: as a solution, we fill the missing values with the mean(avg) of the column.
    reviews less then this avg (e.g 8.91) would be considered as a "bad review", while reviews above this avg would be considered as a "good review".
"""

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

#! 0. extract the data from csv

raw_csv_data = np.loadtxt('3.0Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1]  # all rows, all columns except the first and last (first for id, last for the target)
targets_all = raw_csv_data[:, -1]  # all rows, only the last column (the target)

#! 1. preprocessing: balance the dataset, divide the data into training, validation, and test, save the data in a tensor friendly format (.npz)
#* 1.1. balance the dataset (# target=1 ~= # target=0)
num_one_targets = int(np.sum(targets_all))  # count how many targets are 1
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

#* 1.2. standardize the inputs and shuffle data

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

shuffled_indices = np.arange(scaled_inputs.shape[0])  # create an array of indices
np.random.shuffle(shuffled_indices)  # shuffle the indices

shuffled_inputs = scaled_inputs[shuffled_indices]  # shuffle the inputs
shuffled_targets = targets_equal_priors[shuffled_indices]  # shuffle the targets

#* 1.3. divide the data into training, validation, and test sets (e.g. 70% train, 15% val, 15% test)
samples_count = shuffled_inputs.shape[0]  # number of samples
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)  # print the number of targets=1, total samples, and the ratio of targets=1 in the training set
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)  # same for validation set
print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)  # same for test set

#* 1.4. save the three datasets in .npz

np.savez('3.01Audiobooks_data_train', inputs=train_inputs, targets=train_targets)  # save the training set
np.savez('3.01Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)  # save the validation set
np.savez('3.01Audiobooks_data_test', inputs=test_inputs, targets=test_targets)  # save the test set

#! 2. create the ml algorithm
#* 2.1 data
# let's create a temporary variable "npz", where we will store each of the three Audiobooks datasets
npz = np.load('3.01Audiobooks_data_train.npz')

# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(int)

npz = np.load('3.01Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(float), npz['targets'].astype(int)

npz = np.load('3.01Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(float), npz['targets'].astype(int)

#* 2.2 model
input_size = 10 # 
output_size = 2 # 0 or 1
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax') # use softmax act. func. for output layer of classifier models
])

# optimizer and the loss func. settings
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# hyperparameters
batch_size = 100
max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2) # earlystopping to prevent overfitting

model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=max_epochs, callbacks=[early_stopping], validation_data=(validation_inputs, validation_targets), verbose=2)

#* 2.3. test the model (evaluation)
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=2)
print('\nTest loss: {0: .2f}. Test Accuracy: {1: .2f}%'.format(test_loss, test_accuracy * 100))