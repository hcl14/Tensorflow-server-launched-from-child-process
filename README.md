# Tensorflow-server-launched-from-child-process

## Different ways to call tensorflow model from multiprocessing process without huge memory and time overhead.



Problematic example using `tf.train.Server`, which was subject of the question (working):
----------------------------------------------------------------------------------------

Minimal reproducible example for the question https://stackoverflow.com/questions/52700621/

Here I launch Tensorflow server and open session from spawned child process in Python. 

I want to understand, why I need to re-initialize global variables each time and cannot just lock once initialized graph for that. Also I have weird CUDA OOM error in this program,

In addition, I seek for more efficient ways to reduce computational and memory overhead when using this scenario.


Files:

`keras_simple_gru_model.py`  - train toy GRU model in keras.

`model-best.h5` - saved weights to be loaded in Tensorflow.

`tf_load.py` - main example showing my problem. See `PROBLEM` word in comments.

Tensorflow versions tested: 1.2, 1.11-rc1

If you have import error, you may need to change imports: `tensorflow.contrib.keras.***`, to `tensorflow.keras.***` of something.



Good working examples:
------------------

1.`tf_load_thread.py` - shows that problem is not existent for threads, as `tf.Session()` is thread-safe

2. `tf_load_eager.py` - First solution to the problem, suggested by Allen Lavoie, which uses tensorflow eager execution (incompatible with TF 1.2). This requries saving model without optimizer (`model_converter_for_eager.py`, `model_without_opt.h5`). After that model starts to require float inputs instead of integers. 


3. Second solution: custom tensorflow server.

`tf_server.py` - custom server, where tensorflow is independently loaded for multiple processes (needs manual memory management on GPU)

`tf_load_client.py` - client (main program) which makes requests to TF server. 
