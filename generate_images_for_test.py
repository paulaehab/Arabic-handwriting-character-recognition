# Import main libraries necessary for this project
from gettext import install
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import libraries needed for reading image and processing it
import csv
from PIL import Image
from scipy.ndimage import rotate

# Load Test data
# Testing letters images url
url3 ="https://drive.google.com/file/d/1EPbu952e7oCxcL6l38_HxqF_y_ajh42G/view?usp=sharing"
letters_testing_images_file_path = 'https://drive.google.com/uc?export=download&id='+url3.split('/')[-2]
# Testing letters labels url
url4 ="https://drive.google.com/file/d/1SkoNNi_1HVhuS7CSi9OC8UtJaIUEpyzZ/view?usp=sharing"
letters_testing_labels_file_path = 'https://drive.google.com/uc?export=download&id='+url4.split('/')[-2]


# Loading dataset into dataframes
testing_letters = pd.read_csv(letters_testing_images_file_path, compression='zip', header=None)
testing_letters_labels = pd.read_csv(letters_testing_labels_file_path, compression='zip', header=None)

# print statistics about the dataset
print("There are %d testing arabic letter images of 64x64 pixels." %testing_letters.shape[0])
testing_letters.head()

def convert_pixels_to_image(image_values,n, display=False):
    
# put image values into numpy array
    image_array = np.asarray(image_values)
#reshape the image to be 64*64 image
    image_array = image_array.reshape(64, 64).astype('uint8')
# flip the images as tha dataset is reflected
    image_array = np.flip(image_array, 0)
# rotate the image -90 degree to appear right
    image_array = rotate(image_array, -90)
# from the image from thge array of pixels 
    new_image = Image.fromarray(image_array)
    if display == True:
        new_image.show()
    new_image.save('./images/'+str(n)+'.jpg')
    return new_image
for i in range(0,60):
    convert_pixels_to_image(testing_letters.loc[i],i)


