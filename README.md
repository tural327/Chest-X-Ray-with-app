# Chest-X-Ray-with-app

Data downloaded from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Application Details
- App developed Windows 10
- Python 3.9

**Python libraries I uesd**
- Tensorflow
- OpenCV
- Pyqt6
- glob
- matplotlib

## Data Exploration

It was simple one we had just 2 classes (Normal and Pneumonia) and by using mathpolit library 10 random image displayed

```python

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
 ```
 
 ![](https://github.com/tural327/test/blob/main/chest/nor_p.gif)

 ## Buildig model ##
 
 As input I used 80% of data for training and 20% for validation and each image resized 240x240
 
 
 
 ```python
 train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(240, 240),
  batch_size=32)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(240, 240),
  batch_size=32)
  
 ```
I used convolutional neural network for my classification 

```python
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
  
 ```
 
 Results was good:
 
 ![](https://github.com/tural327/test/blob/main/chest/loss.PNG)
 ![](https://github.com/tural327/test/blob/main/chest/acc.PNG)
 
 Model saved named by "my_model.h5"
 
 ## Desktop app ##
 
 I used Pyqt6 for make simple GUI and I tried to do user friendly so you need just drag your image and click button
 
  ![](https://github.com/tural327/test/blob/main/chest/res_app.gif)
 
