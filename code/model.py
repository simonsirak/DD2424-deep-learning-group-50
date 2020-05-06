import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from load_data import DataLoader

loader = DataLoader(split="100%")
loader.normalizeAllData()

train_ds = loader.getTrainData()
train_ds = train_ds.shuffle(128)

val_ds = loader.getValData()

test_ds = loader.getTestData()

class MyModel(tf.keras.Model):

    def __init__(self, num_classes=30):
        super(MyModel, self).__init__(name="baseWithPyramidPooling")

        # base model
        self.base_model = tf.keras.applications.resnet.ResNet50(input_shape=(1024, 2048, 3), weights="imagenet", include_top=False)
        self.base_model.trainable = True

        # upsample (just temporary)
        self.up0 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")

        # pyramid pooling module
        self.ppool1 = tf.keras.layers.MaxPooling2D((128, 256), strides=(1, 1))
        self.ppool2 = tf.keras.layers.MaxPooling2D((127, 255), strides=(1, 1))
        self.ppool3 = tf.keras.layers.MaxPooling2D((126, 254), strides=(1, 1))
        self.ppool6 = tf.keras.layers.MaxPooling2D((123, 251), strides=(1, 1))

        self.pconv = tf.keras.layers.Conv2D(1, (1, 1), activation="relu")

        # concatenation layer
        self.concat0 = tf.keras.layers.Concatenate()

        # convolution layer (temporary)
        self.conv0 = tf.keras.layers.Conv2D(num_classes, (1, 1), activation="relu")

        # upsample layer (temporary)
        self.up1 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")

        # softmax
        self.soft0 = tf.keras.layers.Softmax()

    def call(self, input):
      
        feature_map = self.base_model(input)
        feature_map_upsampled = self.up0(feature_map)
        
        bin_pool1 = self.ppool1(feature_map_upsampled)
        bin_pool2 = self.ppool2(feature_map_upsampled)
        bin_pool3 = self.ppool3(feature_map_upsampled)
        bin_pool6 = self.ppool6(feature_map_upsampled)
        
        bin_pool1_conv = self.pconv(bin_pool1)
        bin_pool2_conv = self.pconv(bin_pool2)
        bin_pool3_conv = self.pconv(bin_pool3)
        bin_pool6_conv = self.pconv(bin_pool6)
        
        bin_pool1_conv_upsampled = tf.image.resize(bin_pool1_conv, [128, 256])
        bin_pool2_conv_upsampled = tf.image.resize(bin_pool2_conv, [128, 256])
        bin_pool3_conv_upsampled = tf.image.resize(bin_pool3_conv, [128, 256])
        bin_pool6_conv_upsampled = tf.image.resize(bin_pool6_conv, [128, 256])
        
        feature_map_bins_concat = self.concat0([feature_map_upsampled, bin_pool1_conv_upsampled, bin_pool2_conv_upsampled, bin_pool3_conv_upsampled, bin_pool6_conv_upsampled])
        output_num_classes_depth = self.conv0(feature_map_bins_concat)
        output_upsampled = self.up1(output_num_classes_depth)
        output_softmax = self.soft0(output_upsampled)
        
        return output_softmax
      
model = MyModel(30)
train_ds = train_ds.batch(2)

model.compile(optimizer="SGD", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model.fit(x=train_ds, validation_data=val_ds, epochs=3)
