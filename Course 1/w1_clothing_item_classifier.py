import tensorflow as tf
print(tf.__version__)

# callbacks are used to stop the training process if certain accuracy, loss threshold limits are reached.
class myCallback(tf.keras.callbacks.Callback):
  # wait till the end of epoch
  def on_epoch_end(self, epoch, logs={}):
    # loss or accuracy
    if(logs.get('accuracy')>0.88):
      print("\nReached 88% accuracy so cancelling training!")
      self.model.stop_training = True

# get mnist fashion dataset
mnist = tf.keras.datasets.fashion_mnist

# train, test split mnist dataset
# (x_train, y_train), (x_test, y_test)
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

callbacks = myCallback()

# normalize pixel values
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Create model (input, hidden, output)

# Sequential: That defines a SEQUENCE of layers in the neural network
# Flatten: Flatten takes the square image data (N-dimensional set) and turns it into a 1 dimensional set.
# Dense: Adds a layer of neurons
# Each layer of neurons need an activation function to tell them what to do.
# Relu effectively means "If X>0 return X, else return 0" -- so what it does is it only passes values 0 or greater to the next layer in the network.
# Softmax takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0]
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Build model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model with training data
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# evaluate the model testing data
model.evaluate(test_images, test_labels)

# predict test images
classifications = model.predict(test_images)

print(classifications[5])
print(test_labels[5])

# Adding more Neurons we have to do more calculations, slowing down the process, 
# but in this case they have a good impact -- we do get more accurate. 
# That doesn't mean it's always a case of 'more is better', 
# you can hit the law of diminishing returns very quickly.


