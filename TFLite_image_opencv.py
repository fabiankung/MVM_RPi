# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:31:23 2022

@author: user
"""

import os
import numpy as np
import tensorflow as tf
import cv2

TFLITE_MODEL_DIR = '.\TfLite_model'
PATH_TO_TFLITE_MODEL = os.path.join(TFLITE_MODEL_DIR,'model.tflite')

#Set the width and height of the input image in pixels.
_imgwidth = 160
_imgheight = 120

#Set the region of interest start point and size.
#Note: The coordinate (0,0) starts at top left hand corner of the image frame.
_roi_startx = 30
_roi_starty = 71
_roi_width = 100
_roi_height = 37


# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(PATH_TO_TFLITE_MODEL)

# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# Optional, show the format for input.
input_details = interpreter.get_input_details()
# input_details is a dictionary containing the details of the input
# to this neural network.
print(input_details[0])
print(input_details[0]['shape'])
# Now print the signature input and output names.
print(interpreter.get_signature_list())

# Read a grayscale image of dimension 160x120x1.
image_ori = cv2.imread('Grayscale160x120.bmp') 

# Crop out region-of-interest (ROI)
imgcrop = image_ori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width] 
# Convert to grayscale
imgcropgray = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2GRAY)         # Convert to grayscale.

# Normalize each pixel value to floating point, between 0.0 to +1.0
# NOTE: This must follows the original mean and standard deviation 
# values used in the TF model. Need to refer to the model pipeline.
# In Tensorflow, the normalization is done by the detection_model.preprocess(image) 
# method. In TensorFlow lite we have to do this explicitly. 
imgcropgraynorm = imgcropgray/256.0
      
# --- Method 1 to prepare the input ---
#test = np.expand_dims(imgcropgraynorm,0)
#test = np.expand_dims(test,-1)
#test = np.expand_dims(imgcropgraynorm,(0,-1)) # change the shape from (37,100) to (1,37,100,1), 
                                              # to meet the requirement of tflite interpreter
                                              # input format. Also datatype is float32, see
                                              # the output of print(input_details[0])
#input_tensor = tf.convert_to_tensor(test, dtype=tf.float32)

# --- Method 2 to prepare the input, only using numpy ---
input_tensor = np.asarray(np.expand_dims(imgcropgraynorm,(0,-1)), dtype = np.float32)

output = my_signature(conv2d_input = input_tensor)  # Perform inference on the input. The input and 
                                                    # output names can
                                                    # be obtained from interpreter.get_signature_list()

output1 = np.squeeze(output['dense_1'])             # Remove 1 dimension from the output. The output 
                                                    # parameters are packed into a dictionary. With 
                                                    # the name 'dense_1' to access the output layer. 
result = np.argmax(output1) 

print("Result is %d" % result)
cv2.imshow("Result",image_ori)           # Display the image frame.

key = cv2.waitKey(0) # Wait for user input, wait forever.
cv2.destroyAllWindows()