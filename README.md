# Tensorflow-server-launched-from-child-process
Minimal reproducible example for the question https://stackoverflow.com/questions/52700621/

Here I launch Tensorflow server and open session from spawned child process in Python. 

I want to understand, why I need to re-initialize global variables each time and cannot just lock once initialized graph for that. Also I have weird CUDA OOM error in this program,

In addition, I seek for more efficient ways to reduce computational and memory overhead when using this scenario.


Files:

`keras_simple_gru_model.py`  - train toy GRU model in keras.

`model-best.h5` - saved weights to be loaded in Tensorflow.

`tf_load.py` - main example showing my problem. See `PROBLEM` word in comments.
