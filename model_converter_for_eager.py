from keras.models import load_model, Model
from keras.layers import Input, Dense

from keras.models import load_model
model = load_model('model-best.h5')

# save without optimizer
#https://github.com/keras-team/keras/issues/8136

model.save('model_without_opt.h5', include_optimizer=False)
