#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

def load_model():
    # Assuming the model is a TensorFlow/Keras model
    model = tf.keras.models.load_model('model_directory/')
    return model

def preprocess_input(frame):
    # Example preprocessing function
    frame_processed = frame / 255.0  # Normalize the frame
    return frame_processed

