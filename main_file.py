import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import glob
import cv2


## Data Explorer Desktop/project/Chest-X-Ray-with-app/
NORMAL = glob.glob("train/NORMAL/*")

PNEUMONIA =  glob.glob("train/PNEUMONIA/*")



for i in range(10):
    nor = random.randint(0,len(NORMAL))
    pne = random.randint(0,len(PNEUMONIA))
    
    img_normal = cv2.imread(NORMAL[nor])
    img_p = cv2.imread(PNEUMONIA[nor])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_normal)
    ax.set_title('Normal')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_p)
    imgplot.set_clim(0.0, 0.7)
    ax.set_title('PNEUMONIA')

############  Train ###############################

# 240x240 will be my image size 
batch_size = 32
img_height = 240
img_width = 240


## training dir...
data_dir = "train"

# 80% of image will be train our model and 20 % for validation
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


data_augmentation = tf.keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# model building 
class my_class(tf.keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(my_class, self).__init__()
        # -----------------------------------------------------------
        # data_augmentation layer using to get best model beacause loss was not good 
        self.aug = data_augmentation
        self.in_ly = layers.experimental.preprocessing.Rescaling(1./255)
        # conv layers -- so I used 3 convolution layer my model
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
        x = self.aug(inputs)
        x = self.in_ly(x)
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

plt.figure(figsize=(20,10))
plt.plot(history.history['loss'],color='green',linewidth=3,label="Loss")
plt.plot(history.history['val_loss'],color='green',linewidth=1,linestyle="--",label="Val Loss")
plt.title("Model Loss",fontsize=18)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.rcParams.update({'font.size': 20})
plt.legend(loc=3, prop={'size': 20})




test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "test/",
  validation_split=0.01,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)




classifier.evaluate(test_ds)

classifier.save("my_model")






