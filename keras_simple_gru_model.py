# Source: 
#https://github.com/buomsoo-kim/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/4-Advanced-RNN-3/4-2-gru.py

from keras.datasets import imdb
from keras.layers import GRU, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

num_words = 30000
maxlen = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)

# pad the sequences with zeros 
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train, maxlen = maxlen, padding = 'post')
X_test = pad_sequences(X_test, maxlen = maxlen, padding = 'post')

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

print(X_test.shape)

def gru_model():
    model = Sequential()
    model.add(GRU(50, input_shape = (300,1), return_sequences = True))
    model.add(GRU(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_crossentropy', 'accuracy'])
    return model
    
model = gru_model()

checkpoint = ModelCheckpoint('model-best.h5', verbose=1, monitor='val_loss',save_best_only=1, mode='auto')

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 5, callbacks=[checkpoint], verbose = 1)
