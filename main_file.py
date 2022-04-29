# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import random
import glob
import cv2



############  Train ###############################

img_normal = cv2.imread("Desktop/project/Chest-X-Ray-with-app/train/NORMAL\IM-0115-0001.jpeg")
img_p = cv2.imread("Desktop/project/Chest-X-Ray-with-app/train/NORMAL\IM-0115-0001.jpeg")


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img_normal)
ax.set_title('Normal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img_p)
imgplot.set_clim(0.0, 0.7)
ax.set_title('PNEUMONIA')


batch_size = 32
img_height = 180
img_width = 180



data_dir = "Desktop/project/Chest-X-Ray-with-app/train"


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)




val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class my_class(tf.keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(my_class, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # input
        self.in_ly = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3))
        # conv layers
        self.conv1 = layers.Conv2D(16, 4, padding='same', activation='relu')
        self.max1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 4, padding='same', activation='relu')
        self.max2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(64, 4, padding='same', activation='relu')
        self.max3 = layers.MaxPooling2D()
        # Flatten Layer
        self.flt = layers.Flatten()
        self.den1 = layers.Dense(128, activation='relu')

        #end
        self.den2 = layers.Dense(1, activation='sigmoid')

    # Forward pass of model - order does matter.
    def call(self, inputs):
        x = self.in_ly(inputs)
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.flt(x)
        x = self.den1(x)
        return self.den2(x) # Return results of Output Layer


classifier = my_class()
classifier.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])


history = classifier.fit(train_ds,validation_data=val_ds,epochs=20)


plt.figure(figsize=(20,10))
plt.plot(history.history['accuracy'],color='green',linewidth=3,label="Accurancy")
plt.plot(history.history['val_accuracy'],color='green',linewidth=1,linestyle="--",label="Val Accurancy")
plt.title("Model Accurancy",fontsize=18)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Accurancy",fontsize=18)
plt.rcParams.update({'font.size': 20})
plt.legend(loc=4, prop={'size': 20})