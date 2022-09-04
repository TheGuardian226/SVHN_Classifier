import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, MaxPool2D, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load the dataset (Can be downloaded from the link above)
train = loadmat('data/train_32x32.mat')
test = loadmat('data/test_32x32.mat')

# Extract the images and labels from the dictionary object
X_train = train['X']
X_test = test['X']
y_train = train['y']
y_test = test['y']

# Rearrange the dimensions of the image data
X_train = np.moveaxis(X_train, -1, 0)
X_test = np.moveaxis(X_test, -1, 0)

# Convert images to grayscale, normalize to [0,1] and reshape
X_train_grey = np.mean(X_train, 3).reshape(73257, 32, 32,1)/255
X_test_grey = np.mean(X_test, 3).reshape(26032, 32, 32,1)/255
X_train_plt = np.mean(X_train, 3)

# One-hot encode the labels
encoder=OneHotEncoder().fit(y_train)
y_train_encoded = encoder.transform(y_train).toarray()
y_test_encoded = encoder.transform(y_test).toarray()


### Model 1 - Multi-layer perceptron ###
# Define the callbacks
best_val_acc_path = 'best_val_acc/checkpoint'
best_val_acc = ModelCheckpoint(filepath = best_val_acc_path, save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max')
earlystopping = EarlyStopping(patience = 3, monitor='loss')

# Define the model
model_1 = Sequential([
    Flatten(input_shape=X_train[0].shape),
    Dense(512, activation = 'relu', ),
    Dense(128, activation = 'relu'),
    Dense(128, activation='relu'),
    Dense(10, activation ='softmax')
])

# Compile the model
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_1 = model_1.fit(X_train, y_train_encoded, epochs=20, validation_data = (X_test, y_test_encoded), callbacks=[best_val_acc, earlystopping], batch_size=128)

### Model 2 - Convolutional neural network ###
# Define the callbacks
best_val_acc_path_2='CNN/best'
best_val_acc=ModelCheckpoint(filepath = best_val_acc_path_2, save_best_only=True, monitor='val_accuracy', save_weights_only=True, mode='max')
earlystopping = EarlyStopping(monitor='loss', patience=5, verbose=1)

# Define the model
model_2 = Sequential([
    Conv2D(filters= 16, kernel_size= 3, activation='relu', input_shape=X_train[0].shape),
    MaxPool2D(pool_size= (3,3)),
    Conv2D(filters= 32, kernel_size = 3, padding='valid', activation='relu'),
    MaxPool2D(pool_size = (2,2), strides = 2),
    BatchNormalization(),
    Conv2D(filters= 32, kernel_size = 3, padding='valid', strides=2, activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the model
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_2 = model_2.fit(X_train, y_train_encoded, callbacks=[best_val_acc, earlystopping], epochs=20, validation_data=(X_test, y_test_encoded), batch_size=128)