import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import glob
import cv2


## Data Explorer Desktop/project/Chest-X-Ray-with-app/
NORMAL = glob.glob("Desktop/project/Chest-X-Ray-with-app/train/NORMAL/*")

PNEUMONIA =  glob.glob("Desktop/project/Chest-X-Ray-with-app/train/PNEUMONIA/*")



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
data_dir = "Desktop/project/Chest-X-Ray-with-app/train"

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

def create_model():
  model = tf.keras.models.Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])
    
  model.compile(optimizer='adam',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])
  return model

    
model = create_model()
model.summary()



history = model.fit(train_ds,validation_data=val_ds,epochs=20)

model.save("Desktop/project/Chest-X-Ray-with-app/my_model.h5")


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






