# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:31:23 2022

@author: user
"""

import os
import numpy as np
import cv2

TFLITE_MODEL_DIR = '.\TfLite_model'
PATH_TO_TFLITE_MODEL = os.path.join(TFLITE_MODEL_DIR,'model.tflite')

_SHOW_COLOR_IMAGE = False

#Set the width and height of the input image in pixels.
_imgwidth = 160
_imgheight = 120

#Set the region of interest start point and size.
#Note: The coordinate (0,0) starts at top left hand corner of the image frame.
_roi_startx = 30
_roi_starty = 71
_roi_width = 100
_roi_height = 37


# --- For PC ---
import tensorflow as tf
interpreter = tf.lite.Interpreter(PATH_TO_TFLITE_MODEL) # Load the TFLite model in TFLite Interpreter
# === For Raspberry Pi ---
#import tflite_runtime.interpreter as tflite # Use tflite runtime instead of TensorFlow.
#interpreter = tflite.Interpreter(PATH_TO_TFLITE_MODEL)

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

video = cv2.VideoCapture(0) # Open a camera connected to the computer.
video.set(3,2*_imgwidth)   # Set the resolution output from the camera.
video.set(4,2*_imgheight)  # 

# Calculate the corners for all rectangules that we are going to draw on the image.
pointROIrec1 = (2*_roi_startx,2*_roi_starty)
pointROIrec2 = (2*(_roi_startx + _roi_width),2*(_roi_starty + _roi_height))

interval = np.floor(_roi_width/3)
interval2 = np.floor(2*_roi_width/3)
# Rectangle for label1 (object on left)
pointL1rec1 = (2*(_roi_startx+4),2*(_roi_starty+4))
pointL1rec2 = (2*(_roi_startx +int(interval)-4),2*(_roi_starty + _roi_height-4))
# Rectangle for label2 (object on right)
pointL2rec1 = (2*(_roi_startx+4+int(interval2)),2*(_roi_starty+4))
pointL2rec2 = (2*(_roi_startx+_roi_width-4),2*(_roi_starty+_roi_height-4))
# Rectangle for label3 (object in front)
pointL3rec1 = (2*(_roi_startx+4+int(interval)),2*(_roi_starty+4))
pointL3rec2 = (2*(_roi_startx+int(interval2)-4),2*(_roi_starty+_roi_height-4))
# Rectangle for label4 (object blocking front)
pointL4rec1 = (2*(_roi_startx+4),2*(_roi_starty+4))
pointL4rec2 = (2*(_roi_startx + _roi_width-4),2*(_roi_starty + _roi_height-4))

print(pointL1rec1,pointL1rec1)
if not video.isOpened():            # Check if video source is available.
    print("Cannot open camera or file")
    exit()
    
while True:                         # This is same as while (1) in C.
    successFlag, img = video.read() # Read 1 image frame from video.
    
    if not successFlag:             # Check if image frame is correctly read.
        print("Can't receive frame (stream end?). Exiting ...")    
        break
    
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # Convert to grayscale.
    imggrayresize = cv2.resize(imggray,None,fx=0.5,fy=0.5)  # Resize to 160x120 pixels
    
    # Crop out region-of-interest (ROI)
    imggrayresizecrop = imggrayresize[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width] 

    # Normalize each pixel value to floating point, between 0.0 to +1.0
    # NOTE: This must follows the original mean and standard deviation 
    # values used in the TF model. Need to refer to the model pipeline.
    # In Tensorflow, the normalization is done by the detection_model.preprocess(image) 
    # method. In TensorFlow lite we have to do this explicitly. 
    imggrayresizecropnorm = imggrayresizecrop/256.0                 # Normalized to 32-bits floating points  


    #test = np.expand_dims(imgpgrayresizecropnorm,(0,-1)) # change the shape from (37,100) to (1,37,100,1), 
                                              # to meet the requirement of tflite interpreter
                                              # input format. Also datatype is float32, see
                                              # the output of print(input_details[0])
    # --- Method 1 using tf.convert_to_tensor to make a tensor from the numpy array ---
    #input_tensor = tf.convert_to_tensor(test, dtype=tf.float32)

    # --- Method 2 to prepare the input, only using numpy ---
    input_tensor = np.asarray(np.expand_dims(imggrayresizecropnorm,(0,-1)), dtype = np.float32)

    output = my_signature(conv2d_input = input_tensor)  # Perform inference on the input. The input and 
                                                    # output names can
                                                    # be obtained from interpreter.get_signature_list()

    output1 = np.squeeze(output['dense_1'])         # Remove 1 dimension from the output. The output 
                                                    # parameters are packed into a dictionary. With 
                                                    # the name 'dense_1' to access the output layer. 
    result = np.argmax(output1) 

    if _SHOW_COLOR_IMAGE == True:
         # Draw ROI border on image
        cv2.rectangle(img,pointROIrec1,pointROIrec2,(255,0,0), thickness=2)    
        # Draw rectangle for Label 1 to 4 in ROI    
        if result == 1:
            cv2.rectangle(img,pointL1rec1,pointL1rec2,(255,255,0), thickness=2)
        elif result == 2:
            cv2.rectangle(img,pointL2rec1,pointL2rec2,(255,255,0), thickness=2)
        elif result == 3:
            cv2.rectangle(img,pointL3rec1,pointL3rec2,(255,255,0), thickness=2)
        elif result == 4:
            cv2.rectangle(img,pointL4rec1,pointL4rec2,(255,255,0), thickness=2)       
            
        cv2.imshow("Video",img)           # Display the image frame.
    else:
        # Draw ROI border on image
        cv2.rectangle(imggray,pointROIrec1,pointROIrec2,255, thickness=2)    
        # Draw rectangle for Label 1 to 4 in ROI    
        if result == 1:
            cv2.rectangle(imggray,pointL1rec1,pointL1rec2,255, thickness=2)
        elif result == 2:
            cv2.rectangle(imggray,pointL2rec1,pointL2rec2,255, thickness=2)
        elif result == 3:
            cv2.rectangle(imggray,pointL3rec1,pointL3rec2,255, thickness=2)
        elif result == 4:
            cv2.rectangle(imggray,pointL4rec1,pointL4rec2,255, thickness=2)
    
        cv2.imshow("Video",imggray)           # Display the image frame.

    if cv2.waitKey(1) & 0xFF == ord('q'): # Note: built-in function ord() scans the 
          break                           # keyboard for 1 msec, returns the integer value                                            
                                          # of a unicode character. Here we compare user key
                                          # press with 'q' 
# When everything done, release the capture resources.
video.release()                                          
cv2.destroyAllWindows()