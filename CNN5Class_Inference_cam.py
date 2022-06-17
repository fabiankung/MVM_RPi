# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:59:13 2022

@author: Fabian Kung
"""

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
#import os
import cv2

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



# Load keras model in H5 format for inferencing
#mymodel = tf.keras.models.load_model('CNN5Class_Predict_model.h5')

# Load model in Keras SavedModel format for inferencing. 
mymodel = tf.keras.models.load_model('./Exported_model_keras')
# NOTE: There are 2 kind of SavedModel formal, using low-level tf.saved_model.save() API
# and using high-level tf.keras.models.save(). So make sure you use the correct load_model()!


"""
#image = plt.imread('ImgColorBar.bmp',format = 'BMP')  # Load BMP image to be analyzed by the CNN.
image = plt.imread('4.bmp',format = 'BMP')
image = image/256.0  # Normalized to between 0 to 1.0.
#Extract only 1 channel of the RGB data, assign to 2D array
imgori = image[0:_imgheight,0:_imgwidth,0]
#Crop the 2D array to only the region of interest
imgcrop = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]

prediction = mymodel.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))

prediction1 = np.squeeze(prediction)

# The following steps basically find the index with largest probability
i = 0
pmax = 0.0
result = 0
for p in prediction1:
    if p > pmax:
        pmax = p
        result = i
    i += 1
        
print(prediction1)
print('result is ' + str(result))
"""
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
    imggrayresizenorm = imggrayresize/256.0                 # Normalized to 32-bits floating points
    
    # Crop out region-of-interest (ROI)
    imggrayresizenormcrop = imggrayresizenorm[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width] 

    # Make prediction on current frame.
    prediction = mymodel.predict(imggrayresizenormcrop.reshape(1,_roi_height,_roi_width,1))
    prediction1 = np.squeeze(prediction)   # Remove 1 dimension from the output.
    '''
    # The following steps basically find the index with largest probability
    i = 0
    pmax = 0.0
    result = 0
    for p in prediction1:
        if p > pmax:
            pmax = p
            result = i
        i += 1
    '''
    # Find the index with largest probability value.
    result = np.argmax(prediction1)    
    
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
    