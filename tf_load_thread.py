
# I am getting CUDA OOM errors when called from process, which is strange
import os
os.environ["CUDA_VISIBLE_DEVICES"]= ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue
import tensorflow as tf

import global_vars

# load data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 100)

# global variable to store finalized graph
g = {} 
NN_MODEL_FILE = 'model-best.h5'
    
        
# model initializer
def init_model_for_thread(example):
    import tensorflow as tf
    from tensorflow.contrib.keras import models
    # create tensorflow graph
    graph = tf.get_default_graph()
    with graph.as_default():
        
        sess = tf.Session() #tf.contrib.keras.backend.get_session()
        
        sess.run(tf.global_variables_initializer())
        
        NN_MODEL = models.load_model(NN_MODEL_FILE)
        
        # global variable with all the data to be shared
        model = {}

        model["NN_MODEL"] = NN_MODEL
        
        # dummy run to create graph        
        x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                         [example],
                         maxlen=300,
                         padding = 'post'
                         )
        x = x.reshape(x.shape + (1,))
        
        y = NN_MODEL.predict(x)
        print('Testing model inference: {}'.format(y))
        
        # PROBLEM: I want, but cannot lock graph, as child process 
        # wants to run its own tf.global_variables_initializer()
        graph.finalize()

        model["GRAPH"] = graph
        return model

# caling model from process
def function_which_uses_inference_model(some_data):
    # access global variable which is supposed to be read-only in fork mode
    model = g["model"]["NN_MODEL"]
    graph = g["model"]["GRAPH"]
    
    
    print('Making inference')
    
    # It seems that kears uses the same session across threads
    
    # finally, make a call to server:
    # the following throws uninitialized variables error:
    # sess= tf.Session(graph=graph)
    # with sess.as_default():
    
    # the following seems to use one session across threads:
    with graph.as_default():
        x = tf.contrib.keras.preprocessing.sequence.pad_sequences(
                            some_data,
                            maxlen=300,
                            padding = 'post'
                            )
        print('Making inference 1')
        x = x.reshape(x.shape + (1,))
        print('Making inference 2')
        y = model.predict(x)
        print('returning data')
        return y


    
    
if __name__ == "__main__":
    # store model to be accessible for all modules
    # (simplified)
    model = init_model_for_thread(X_test[0])
    g["model"] = model
    print('Model initialized')
    
    # in my real example it's a big read-only dictionary
    some_data = [X_test[1]]
    
    import time
    time.sleep(5) # wait until tf server is up for sure
    
    
    
    # Spawn the thread
    def foo(q):
        result = function_which_uses_inference_model(some_data) 
        q.put(result)
        return # I've read it is essential for destroying local variables
    q = Queue()
    # run two concurrent threads to make sure session is not the problem
    p1 = Thread(target=foo,args=(q,))
    p2 = Thread(target=foo,args=(q,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    result1 = q.get() # retrieve data
    print('Thread finished: {}'.format(result1))
    
    result2 = q.get() # retrieve data
    print('Thread finished: {}'.format(result2))
    
    
