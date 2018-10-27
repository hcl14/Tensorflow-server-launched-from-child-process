
# I am getting CUDA OOM errors when called from process, which is strange
import os
os.environ["CUDA_VISIBLE_DEVICES"]= ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from multiprocessing import Process, Queue
import tensorflow as tf
from tensorflow.contrib.keras import models
tf.enable_eager_execution()

import global_vars

# load data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 100)

g = {} 
NN_MODEL_FILE = 'model_without_opt.h5'
    
            
# model initializer
def init_model_for_process(example):
    
    # restore in eager execution
    # https://github.com/keras-team/keras/issues/8136

        
    NN_MODEL = models.load_model(NN_MODEL_FILE)
    
    # Need to add some optimizer to make model compiled
    # Only TF native optimizers are supported in Eager mode.
    NN_MODEL.compile(optimizer=tf.train.GradientDescentOptimizer(0.01), loss='mean_squared_error')

    
    # global variable with all the data to be shared
    model = {}

    model["NN_MODEL"] = NN_MODEL
    
    '''
    _________________________________________________________________                                                                                                  
    Layer (type)                 Output Shape              Param #                                                                                                     
    =================================================================                                                                                                  
    gru_1 (GRU)                  (None, 300, 50)           7800                                                                                                        
    _________________________________________________________________                                                                                                  
    gru_2 (GRU)                  (None, 1)                 156                                                                                                         
    _________________________________________________________________                                                                                                  
    activation_1 (Activation)    (None, 1)                 0                                                                                                           
    =================================================================                                                                                                  
    Total params: 7,956                                                                                                                                                
    Trainable params: 7,956                                                                                                                                            
    Non-trainable params: 0                                                                                                                                            
    _________________________________________________________________  
    '''
        
    # Testing
    x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                         [example],
                         maxlen=300,
                         padding = 'post'
                         )
    # now it wants float somewhy
    x = np.array(x.reshape(x.shape + (1,)), dtype=float)

        
    y = NN_MODEL.predict(x)
    print('Testing model inference: {}'.format(y))
        
    
    return model

# caling model from process
def function_which_uses_inference_model(some_data):
    # access global variable which is supposed to be read-only in fork mode
    model = g["model"]["NN_MODEL"]
    
    print('Making inference')
    # finally, make a call to server:
    x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                            some_data,
                            maxlen=300,
                            padding = 'post'
                            )
    print('Making inference 1')
    x = np.array(x.reshape(x.shape + (1,)), dtype=float)
    print('Making inference 2')
    y = model.predict(x)
    print('returning data')
    return y


    
    
if __name__ == "__main__":
    
    # store model to be accessible for all modules
    # (simplified)
    model = init_model_for_process(X_test[0])
    g["model"] = model
    print('Model initialized')
    
    # in my real example it's a big read-only dictionary
    some_data = [X_test[1]]
    
    
    
    
    # Spawn the process
    def foo(q):
        result = function_which_uses_inference_model(some_data) 
        q.put(result)
        return # I've read it is essential for destroying local variables
    q = Queue()
    p = Process(target=foo,args=(q,))
    p.start()
    p.join()
    result = q.get() # retrieve data
    print('Process finished: {}'.format(result))
    
    
 
