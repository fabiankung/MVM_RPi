# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:31:23 2022

@author: user
"""

import os
import tensorflow as tf

SAVED_MODEL_DIR2 = '.\Exported_model_keras'         # Point to Tensorflow (High-level) SavedModel folder.
SAVED_MODEL_DIR = '.\Exported_model_tf'             # Point to Tensorflow (low-level) SavedModel folder.
EXPORT_MODEL_DIR = '.\TfLite_model'                 # Directory to store Tensorflow lite model. Make sure to create this
                                                    # directory if it is not there!

PATH_TO_EXPORT_MODEL = os.path.join(EXPORT_MODEL_DIR,'model.tflite') # Path and filename of export model.
 
mymodel = tf.keras.models.load_model(SAVED_MODEL_DIR2) # NOTE: 14/6/2022 Somehow I kept getting
                                                       # error from python interpreter when this
                                                       # line is not executed.
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
  
tflite_model = converter.convert()

# Save the model.
with open(PATH_TO_EXPORT_MODEL, 'wb') as f:
  f.write(tflite_model)

#from tflite_support.metadata_writers import object_detector
#from tflite_support.metadata_writers import writer_utils