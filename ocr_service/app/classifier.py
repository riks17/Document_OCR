# This module handles the classification of a document image into one of the known types.

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("models/classifier_model.h5")
labels = ['pan', 'passport', 'voterid_new', 'voterid_old']

def classify_document(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    prediction = model.predict(x)
    predicted_index = np.argmax(prediction)
    return labels[predicted_index]