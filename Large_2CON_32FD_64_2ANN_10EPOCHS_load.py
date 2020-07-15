# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:33:18 2019

@author: hpangam
"""

from keras.models import load_model
filename = 'Large_2CON_32FD_64_2ANN_10EPOCHS.sav'
model = load_model(filename)

#print(training_set.class_indices)


import numpy as np
from keras.preprocessing import image
my_image = image.load_img('test_set/19.jpg', target_size = (64,64,3))
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis = 0)
#result = classifier.predict(my_image)

if model.predict(my_image) >= 0.6 :
    print('DOG')
else:
    print('CAT')

#6, 14, 19, 20