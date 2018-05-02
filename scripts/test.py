from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import visualization
import matplotlib.pyplot as plt

json_file = open('model_config.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model


model.load_weights("model_weights.h5")
img = visualization.view(model, 128, 'conv2d_2', 128)
plt.imshow(img)
plt.show()
