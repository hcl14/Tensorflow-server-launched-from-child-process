# simple tensorflow server using tornado/flask 


from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options


from flask import Flask, request, Response, json, abort, jsonify
import json as json2

from multiprocessing import Process

import numpy as np

# global variables, will be overwritten for each process
global_vars = {}

host = '0.0.0.0'
port = '2222'
n_servers = 3 # there will be 3 fully independent instances of tf, hence 3xRAM
# Tensorflow instances run concurrenlty, if all are busy, requests goes into query

NN_MODEL_FILE='model-best.h5'

# Tensorflow tries to allocate all GPU RAM on import
# but we import it multiple times, so CUDA will display OOM
# You need more tuning for Tensorflow (i.e. specify amount of GPU RAM to use)
# if you want to use GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]= ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # reduce tensorflow verbosity



# model initializer
def init_model_for_process():
    
    # importing tensorflow for each process separately
    # import tensorflow as tf
    from tensorflow.contrib.keras import models
    
        
    model = models.load_model(NN_MODEL_FILE)
        
    return model




# create flask application
def create_app():
    
    app = Flask(__name__)
    
    # request should be: {'data': whatever_model_accepts}
    @app.route('/model/predict', methods=['POST'])
    def predict():
        
        prediction = {}
        
        print('%s predicting'% global_vars['id'])
        try:
            # read data from request
            body = json.loads(request.data)
            
            model = global_vars['model']
            
            pred = model.predict(np.array(body['data']))
            
            prediction = {'data':pred.tolist()}
        except Exception as e:
            print(e)
        
        return Response(json.dumps(prediction))

    return app





# create tornado server for process


# you may specify these via command line like so:
# python urls-service.py --port=8888 --address=0.0.0.0 --logging=debug
define("port", default=port, help="run on the given port", type=int)
define("address", default=host, help="run on the given address", type=str)


def run(process_id):
            
    try:
        
        # create tensorflow model for this process
        model = init_model_for_process()
        # write it to process global scope ('fork' mode)
        global_vars['model'] = model
        global_vars['id'] = process_id
        
        app = create_app() 
        
        ioloop = IOLoop()

        
        http_server_api = HTTPServer(WSGIContainer(app))
        # reuse_port allows multiple servers co-exist
        http_server_api.bind(address=options.address, port=options.port, reuse_port=True) 
        http_server_api.start()
        
        print("Server %s started %s:%s" % (process_id, options.address,
                                            options.port))

        ioloop.start()
    except Exception as e:
        print(e)
        



# start processes:
if __name__ == '__main__':
    processes = []
    
    for i in range(0, n_servers):
        p = Process(target=run, args=(str(i),))
        p.daemon = False # we want to spawn child processes
        processes.append(p)
    
    # Run processes:
    for p in processes:
        p.start()
        
    # wait intil all processes end (just to block)
    for p in processes:
        p.join() 
