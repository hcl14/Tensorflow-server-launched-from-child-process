
# I am getting CUDA OOM errors when called from process, which is strange
import os
os.environ["CUDA_VISIBLE_DEVICES"]= ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from multiprocessing import Process, Queue
import tensorflow as tf

import global_vars

# load data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 100)

g = {} 
tf_server_address = 'localhost:2222'
NN_MODEL_FILE = 'model-best.h5'
    
    
# initialize tensorflow server
def start_tf_server():
    import tensorflow as tf
    cluster = tf.train.ClusterSpec({"local": [tf_server_address]})
    server = tf.train.Server(cluster, job_name="local", task_index=0)    
    server.join()
        
# model initializer
def init_model_for_process(example):
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
        #graph.finalize()

        model["GRAPH"] = graph
        return model

# caling model from process
def function_which_uses_inference_model(some_data):
    # access global variable which is supposed to be read-only in fork mode
    model = g["model"]["NN_MODEL"]
    graph = g["model"]["GRAPH"]
    
    # open session to server
    sess = tf.Session('grpc://'+tf_server_address, graph=graph)
    print('Opened server session')
    
    # PROBLEM: and I need to run variables initializer:
    sess.run(tf.global_variables_initializer())
    
    # Seems unnecessary
    # Hangs on tf_session.TF_CloseSession(self._session)
    # when launched from process in Tensorflow 1.11.0-rc1
    # Does not hang in Tensorflow 1.2
    #tf.contrib.keras.backend.set_session(sess)
    
    print('Making inference')
    # finally, make a call to server:
    with sess.as_default():     
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
    # start tf server
    p = Process(target=start_tf_server)
    p.daemon=True
    p.start()
    
    print('Tf server started')
    
    # store model to be accessible for all modules
    # (simplified)
    model = init_model_for_process(X_test[0])
    g["model"] = model
    print('Model initialized')
    
    # in my real example it's a big read-only dictionary
    some_data = [X_test[1]]
    
    import time
    time.sleep(5) # wait until tf server is up for sure
    
    
    
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
    
    
