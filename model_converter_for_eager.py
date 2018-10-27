
from tensorflow.keras import models
#from keras.models import load_model
#model = load_model('model-best.h5')
model = models.load_model('model-best.h5')

# save without optimizer
#https://github.com/keras-team/keras/issues/8136

#model.save('model_without_opt.h5', include_optimizer=False)
models.save_model(model, filepath='model_without_opt.h5', include_optimizer=False)
