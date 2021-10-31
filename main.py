import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

img_array = cv2.imread('/content/drive/MyDrive/Colab/DL/dog_example.jpeg')
#print(img_array.shape)
DATADIR = '/content/drive/MyDrive/Colab/DL/PetImages/'

CATEGORIES = ["Dog"]

path = os.path.join(DATADIR,CATEGORIES[0])
print(path)
print(os.listdir(path))
img = '9119.jpg'
img_path = os.path.join(path,img)
print(img_path)

img_array = cv2.imread(os.path.join(path,img))
plt.imshow(img_array, cmap='gray')  # graph it
plt.show()  # display!"""
IMG_SIZE = 50
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:  # do dogs and cats
  path = os.path.join(DATADIR,category)  # create path to dogs and cats
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')  # graph it
    plt.show()  # display!"""

    break # Enough
  break
training_data = []

def create_training_data():
  for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
      try:
        img_array = cv2.imread(os.path.join(path,img))  # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        training_data.append([new_array, class_num])  # add this to our training_data
      except Exception as e:  # in the interest in keeping the output clean...
        pass
      #except OSError as e:
      #    print("OSErrroBad img most likely", e, os.path.join(path,img))
      #except Exception as e:
      #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))
import random

random.shuffle(training_data)
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open('/content/drive/MyDrive/Colab/DL/X.pickle',"rb")
X = pickle.load(pickle_in)

pickle_in = open('/content/drive/MyDrive/Colab/DL/y.pickle',"rb")
y = pickle.load(pickle_in)
# Verificar
print("X type of variable:"+str(type(X)))
print("X shape:"+str(X.shape))
print("y type of variable:"+str(type(y)))
import numpy as np

y = np.array(y)
print("y New type of variable:"+str(type(y)))

print("Input X variable:"+str(X.shape[1:]))
# Normalizando
X = X/255.0
# CNN
model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(128))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=30, validation_split=0.3)


pickle_in = open('/content/drive/MyDrive/Colab/DL/X.pickle',"rb")
X = pickle.load(pickle_in)

pickle_in = open('/content/drive/MyDrive/Colab/DL/y.pickle',"rb")
y = pickle.load(pickle_in)
# Verificar
print("X type of variable:"+str(type(X)))
print("X shape:"+str(X.shape))
print("y type of variable:"+str(type(y)))
import numpy as np

y = np.array(y)
print("y New type of variable:"+str(type(y)))
# Normalizando
X = X/255.0
# CNN with different values
conv_layers = [1, 2]
layer_sizes = [128,256]
dense_layers = [0,1]

for dense_layer in dense_layers:
  for layer_size in layer_sizes:
    for conv_layer in conv_layers:
      NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
      print(NAME)
      
      model = Sequential()
      
      model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      
      for l in range(conv_layer-1):
        model.add(Conv2D(layer_size, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
      
      model.add(Flatten())

      for _ in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))

      model.add(Dense(1))
      model.add(Activation('sigmoid'))

      #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
      logdir = os.path.join("logs", format(NAME))
      tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

      model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard_callback])
