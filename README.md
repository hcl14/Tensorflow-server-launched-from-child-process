# Tensorflow-server-launched-from-child-process
Minimal reproducible example for the question https://stackoverflow.com/questions/52700621/

Here I launch Tensorflow server and open session from spawned child process in Python. 

I want to understand, why I need to re-initialize global variables each time and cannot just lock once initialized graph for that. Also I have weird CUDA OOM error in this program,

In addition, I seek for more efficient ways to reduce computational and memory overhead when using this scenario.


Files:

`keras_simple_gru_model.py`  - train toy GRU model in keras.

`model-best.h5` - saved weights to be loaded in Tensorflow.

`tf_load.py` - main example showing my problem. See `PROBLEM` word in comments.

Tensorflow versions tested: 1.2, 1.11-rc1


EDIT:

`tf_load_thread.py` - shows that problem is not existent for threads, as `tf.Session()` is thread-safe

`tf_load_eager.py` - First solution to the problem, suggested by Allen Lavoie, which uses tensorflow eager execution. This requries saving model without optimizer (`model_converter_for_eager.py`, `model_without_opt.h5`). After that model starts to require float inputs instead of integers. 
