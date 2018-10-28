# First you need to get server 'tf_server.py' up and running

from tensorflow.contrib.keras import preprocessing
import requests

# load data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 100)


tf_server_addres = 'http://127.0.0.1:2222/model/predict'

# caling model from process
def function_which_uses_inference_model(some_data):
    
    print('Making inference')
    
    x = preprocessing.sequence.pad_sequences(
            some_data,
            maxlen=300,
            padding = 'post'
            )
    x = x.reshape(x.shape + (1,))
    x = x.tolist()
    
    y = None
    # if x is numpy array, use .tolist() 
    # and then construct numpy array on server side
    try:
        response = requests.post(tf_server_addres, json={'data':x})
        
        if response.status_code == 200:
            y = response.json()
        
    except Exception as e:
        print(e)
        
    return y



if __name__ == "__main__":
    
    some_data = [X_test[1]]
    
    result = function_which_uses_inference_model(some_data) 
    print(result)

