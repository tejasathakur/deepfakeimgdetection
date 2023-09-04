import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

def normalize_image(image, labels):
    image = tf.cast(image, tf.float32) / 255.0
    return image, labels

BATCH_SIZE = 32
IMG_SIZE = (256, 256)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
).map(normalize_image).prefetch(buffer_size=tf.data.AUTOTUNE)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
).map(normalize_image).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
).map(normalize_image).prefetch(buffer_size=tf.data.AUTOTUNE)

model_CNN = Sequential([
    Conv2D(filters=8, kernel_size=3, padding='same', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=16, kernel_size=4, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    Flatten(),
    tf.keras.layers.Dropout(0.3),
    Dense(units=64, activation='relu'),
    Dense(units=20, activation='relu'),
    Dense(units=2, activation='softmax')
])

model_CNN.compile(optimizer = tf.keras.optimizers.Adam(),
                loss = 'BinaryCrossentropy',
                metrics=['accuracy'])
hist = model_CNN.fit(train_data,
                    # @markdown Epochs
                    epochs = 1, # @param {type:"slider", min:0, max:70, step:1}
                    validation_data = validation_data,
                    validation_steps = int(0.5 * len(validation_data))
                    )
