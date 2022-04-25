# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import glob
import cv2




############  Train ###############################

train_normal = glob.glob("chest_xray/train/NORMAL/*")
train_p = glob.glob("chest_xray/train/PNEUMONIA/*")

n_normal = random.randint(0, len(train_normal))
p_normal = random.randint(0, len(train_p))

img_normal = cv2.imread(train_normal[n_normal])
img_p = cv2.imread(train_p[p_normal])


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img_normal)
ax.set_title('Normal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img_p)
imgplot.set_clim(0.0, 0.7)
ax.set_title('PNEUMONIA')



