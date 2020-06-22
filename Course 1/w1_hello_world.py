import tensorflow as tf
import numpy as np
from tensorflow import keras

# define model
# one layer, one neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# optimizer (stochastic gradient descent) guesses while trying to minimize the loss (mse)
model.compile(optimizer='sgd', loss='mean_squared_error')

# data set (y=2x-1)
xs = np.array([-4.0, -3.0, -2.0, -1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
ys = np.array([-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype=float)

# train the neural net
model.fit(xs, ys, epochs=500)

# predict
print(model.predict([10.0]))