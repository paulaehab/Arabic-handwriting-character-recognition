import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import rotate

def convert_image_to_pixels(image):
    
    image_sequence = image.getdata()
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

st.title('Arabic_Handwritten_Recognition')
file_up = st.file_uploader("Upload an image", type= ["jpg", "bmp"])
pred_button = st.button("Predict")
if file_up is not None:

    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    

    

    if pred_button:

        model  = tf.keras.models.load_model('my_model.h5')

        image_array = convert_image_to_pixels(image)
        image_array = image_array.reshape(64, 64).astype('uint8')
        image_array = np.flip(image_array, 0)
        # rotate the image -90 degree to appear right
        image_array = rotate(image_array, -90)
        #reshape image to be 64*64
        image_array = image_array.reshape([-1, 64, 64, 1])
        #normalize image
        image_array = image_array.astype('float32')/255

        y_pred = get_predicted_classes(model, image_array)
        y_pred = y_pred[0]
        predicted_label = convert_categorical_label_to_real_label(y_pred)
        st.write("The model predict the letter as the following:  '{}'".format( predicted_label))
        
    

