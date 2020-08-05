#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:05:01 2020

@author: mathew
"""
from tensorflow.keras.optimizers import RMSprop
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
# Directory with our training horse pictures
train_images = os.path.join('/home/mathew/Documents/Tensorflow/seg_train/')

# Directory with our training human pictures
test_images = os.path.join('/home/mathew/Documents/Tensorflow/seg_test/')

print('Total training images:', len(os.listdir(train_images)))
print('Total training images:', len(os.listdir(test_images)))

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
#%%
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    # This is the source directory for training images
    '/home/mathew/Documents/Tensorflow/seg_train',
    target_size=(150, 150),  # All images will be resized to 150x150
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

# Flow training images in batches of 128 using train_datagen generator
test_generator = test_datagen.flow_from_directory(
    # This is the source directory for training images
    '/home/mathew/Documents/Tensorflow/seg_test',
    target_size=(150, 150),  # All images will be resized to 150x150
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')
#%%
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # outputs to the 
    tf.keras.layers.Dense(6, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, 
                    validation_data=test_generator, verbose=1)
