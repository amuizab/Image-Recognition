import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import pyplot

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation
from tensorflow.keras import Model, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


checkpoint_path = 'modelcheck'

#model = create_model()

load_model = tf.keras.models.load_model('modelproject.h5')

load_model.summary()
print('testttt')


img_size = 224
#labels = ['pumpkin', 'tomato', 'watermelon']
labels = ['hayotest']
data=[]

train_images = []
train_labels = []


'''def dataset(pathnya):
    data=[]
    data2=[]
    pathh = os.path.join(pathnya, 'train')
    print(pathh)
    for folder in labels:
        print(folder)
        print('y')
        path = os.path.join(pathh,folder)
        print(path)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image))
            resized_arr = cv2.resize(img, (img_size, img_size))
            #data.append(np.array(resized_arr))
            #print(data)
            data2.append([np.array(resized_arr),folder])
            #print('data dengan lavel:',data2)

    #return data, data2
    return data2



train = dataset('yabacground')

#print(train)

for feature, label in train:
  train_images.append(feature)
  train_labels.append(label)

print(type(train_images))
print(type(train_labels))
cv2.imshow('ss',np.array(train_images[1]))

#print(np.array(train_images).shape)
#print(np.array(train_images[10]).shape)

train_imagess = np.array(train_images)/255
#print(train_imagess.shape)


print(type(train_labels))

#train_labelss = keras.utils.to_categorical(train_labels, 3)

#print(train_labels)

encoder = LabelEncoder()
train_labelss = encoder.fit_transform(train_labels)
print(train_labelss)

X_train, X_test, y_train, y_test = train_test_split(train_imagess, train_labelss, test_size=0.1, random_state=42)
'''

#img = cv2.imread('pumpkin1.jpg')
#resized_arr = cv2.resize(img, (img_size, img_size))

#n = np.array(resized_arr)
#data = np.array(n)/255

#data = resized_arr/255

#print(data.shape)


#print(data.shape)


'''print(X_test.shape)
print(X_test)
print('xtest')

sui = load_model.predict(X_test)
print(sui.shape)
print('sui jalan')
print(sui)
print(sui[1])
cv2.imshow('suiii',sui[1])

print(y_test.shape)
print(y_test)
'''



def dataset(pathnya):
    data=[]
    data2=[]
    pathh = os.path.join(pathnya, 'train')
    print(pathh)
    for folder in labels:
        print(folder)
        print('y')
        path = os.path.join(pathh,folder)
        print(path)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image))
            print('ini img:',img)
            cv2.imshow('imgnya ni',img)
            resized_arr = cv2.resize(img, (img_size, img_size))
            #data.append(np.array(resized_arr))
            #print(data)
            data2.append([np.array(resized_arr),folder])
            #print('data dengan lavel:',data2)

    #return data, data2
    return data2



train = dataset('yatest')

#print(train)

for feature, label in train:
  train_images.append(feature)
  train_labels.append(label)

print(type(train_images))
print(type(train_labels))

print(np.array(train_images).shape)

#print('fiturrrr:',train_images)
#print('labelssss',train_labels)


cv2.imshow('ss',np.array(train_images[0]))

#print(np.array(train_images).shape)
#print(np.array(train_images[10]).shape)

train_imagess = np.array(train_images)/255


sui = load_model.predict(train_imagess)

suinormal = sui*255
print(sui)
print(suinormal)
cv2.imshow('suiii',sui)
cv2.imshow('suiiinormall',suinormal)

