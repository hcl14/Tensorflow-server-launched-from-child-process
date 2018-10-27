import tensorflow as tf
from tensorflow.contrib.keras import models
import numpy as np

tf.enable_eager_execution()

from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 100)

example = X_test[1]



# restore in eager execution
# https://github.com/keras-team/keras/issues/8136

NN_MODEL = models.load_model("model_without_opt.h5")

# Need to add some optimizer to make model compiled
# Only TF native optimizers are supported in Eager mode.
NN_MODEL.compile(optimizer=tf.train.GradientDescentOptimizer(0.01), loss='mean_squared_error')

x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                         [example],
                         maxlen=300,
                         padding = 'post'
                         )
# now it wants float somewhy
x = np.array(x.reshape(x.shape + (1,)), dtype=float)

# returns numpy, strangely. Not tensor
y = NN_MODEL.predict(x)
print('Testing model inference: {}'.format(y)) # [[0.49816433]]


'''
# load old model in non-eager mode and compare results:

NN_MODEL = models.load_model("model-best.h5")

x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                         [example],
                         maxlen=300,
                         padding = 'post'
                         )
x = x.reshape(x.shape + (1,))
        
y = NN_MODEL.predict(x)
print('Testing model inference: {}'.format(y)) # [[0.49816433]]
'''
