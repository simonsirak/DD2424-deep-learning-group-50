import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# load dataset
cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = keras.applications.resnet50.preprocess_input(train_images)
test_images = keras.applications.resnet50.preprocess_input(test_images)


# load pre-trained ResNet50 with freezed parameters
input_tensor = keras.layers.Input(shape=(32, 32, 3))
base_model = keras.applications.resnet50.ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=False)
base_model.trainable = False

# create new model on top of ResNet50
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10),
    keras.layers.Softmax()
])

# train and evaluate
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=3)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

#----------------------------------------

"""cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels, verbose=2)"""