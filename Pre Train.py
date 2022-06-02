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


img_size = 224
labels = ['pumpkin', 'tomato', 'watermelon']
data=[]

train_images = []
train_labels = []

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

#print('fiturrrr:',train_images)
#print('labelssss',train_labels)


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


X_train, X_test, y_train, y_test = train_test_split(train_imagess, train_labelss, test_size=0.3, random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(train_imagess.shape)
print(train_labelss.shape)



'''print(train_labels)
train_labelss = keras.utils.to_categorical(train_labels, 2)


print(train_labelss)'''

'''basemodel = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),
                                                              activation='relu',
                                                              input_shape=train_imagess.shape)])

basemodel.summary()'''



'''for i in range(1):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	print(i)
	# plot raw pixel data
	pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()
'''

model = keras.Sequential()
weight_decay = 0.0005
model_input = (224, 224, 3)

model.add(Conv2D(64, (3, 3), padding='same', input_shape=model_input, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()



opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

file_path = 'modelcheck'
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', mode='max', verbose=1, periode=10)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[checkpoint])

model.save("modelproject.h5")

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.legend()
plt.show()


