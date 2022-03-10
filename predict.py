import argparse
from gettext import install
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import tensorflow as tf
from tensorflow import keras
import csv
from PIL import Image
from scipy.ndimage import rotate
new_model = tf.keras.models.load_model('model.h5')

def convert_image_to_pixels(url):
    an_image = Image.open(str(url))
    image_sequence = an_image.getdata()
    image_array = np.array(image_sequence)
    return image_array

def get_predicted_classes(model, data, labels=None):
  # parameters are the data of image we want to predict 
  image_predictions = model.predict(data)
  #Get the most valued predicted class from the model
  predicted_classes = np.argmax(image_predictions, axis=1)
  #Get the label of the predicted class
  return predicted_classes
def convert_categorical_label_to_real_label(categorical_label):
  #intial empty list
  real_labels = []
  real_labels.extend([x for x in range(10)])

  # Add the Arbaic letters into a list
  real_labels.extend(['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى'])
  return real_labels[categorical_label]
  
 
  
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="image url")
    args = vars(ap.parse_args())
    image_url = args["image"]
   # convert image to array 
    image_array = convert_image_to_pixels(image_url)
    print(image_array.shape)
    image_array = image_array.reshape(64, 64).astype('uint8')

    image_array = np.flip(image_array, 0)
   # rotate the image -90 degree to appear right
    image_array = rotate(image_array, -90)
    #reshape image to be 64*64
    image_array = image_array.reshape([-1, 64, 64, 1])
    #normalize image
    image_array = image_array.astype('float32')/255
   
    y_pred = get_predicted_classes(new_model, image_array)
    y_pred = y_pred[0]
    predicted_label = convert_categorical_label_to_real_label(y_pred)
    print("The model predicted Letter as '{}'".format( predicted_label))

    
    