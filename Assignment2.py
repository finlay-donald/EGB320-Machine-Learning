# CAB320 Assignment Task 2

#https://keras.io/api/applications/mobilenet/#mobilenetv2-function
#https://keras.io/guides/transfer_learning/


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
import os


if __name__ == "__main__":
    pass
    
    def my_team():
        print(["Finlay Donald", "n10469265"], ["Jacob Minehan", "n10467581"], ["Ethan Schomberg", "n10470247"])
    
    img = plt.imread("./small_flower_dataset/daisy/5794839_200acd910c_n.jpg")
    plt.imshow(img)
    
    
    #Initiate the base model with pre-trained waeights
    base_model = tf.keras.applications.MobileNetV2(
        alpha=1.0,
        include_top=False,
        weights="imagenet"
    )
    
    #Freeze the base model
    base_model.trainable = False
    
    #Make sure the model is running on interface mode
    inputs = keras.Input()
    
    x = base_model(inputs, training = False)
    
    x = keras.layers.GlobalAveragePooling()(x)
    
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
    
    #Loading in the new dataset to train the model on
    new_dataset = plt.imread("./small_flower_dataset/daisy/5794839_200acd910c_n.jpg")
    
    model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
    
    
    ##################################################################
    #Fine turning
    
    # Unfreeze the base model
    base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    # Train end-to-end. Be careful to stop before you overfit!
    model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)
    
    
    
    





