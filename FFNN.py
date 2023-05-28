import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import warnings

def build_ffnn(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=64, activation='relu', name='Hidden1')(input_layer)
    hidden_layer2 = tfkl.Dense(units=64, activation='relu', name='Hidden2')(hidden_layer1)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output')(hidden_layer2)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='FFNN')

    # Compile the model
    loss = tfk.losses.MeanSquaredError()
    learning_rate = 0.2
    optimizer = tfk.optimizers.SGD(learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    # Return the model
    return model

warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers

# Constants
seed = 42
batch_size = 64
epochs = 150

#Code

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

data = load_boston()

boston_dataset = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=['MEDV'])
boston_dataset.describe()

X_train, X_test, y_train, y_test = train_test_split(boston_dataset, target, test_size = 0.2, random_state=seed)

# Normalize both features and target
max_df = X_train.max()
min_df = X_train.min()
max_t = y_train.max()
min_t = y_train.min()

X_train = (X_train - min_df)/(max_df - min_df)
y_train = (y_train - min_t)/(max_t - min_t)

input_shape = X_train.shape[1:]

print("The input shape is: ", input_shape)

ffnn = build_ffnn(input_shape)
ffnn.summary()

history = ffnn.fit(
    x = X_train,
    y = y_train, 
    batch_size = batch_size,
    epochs = epochs
).history

plt.figure(figsize=(15,5))
plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
plt.title('Loss')
plt.legend()
plt.grid(alpha=.3)
plt.show()