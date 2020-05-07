import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from load_data import DataLoader

loader = DataLoader(dataset_name='cityscapes', split="100")
loader.normalizeAllData()

train_ds = loader.getTrainData()
# train_ds = train_ds.shuffle(128)

val_ds = loader.getValData()

test_ds = loader.getTestData()

class MyModel(tf.keras.Model):

    def __init__(self, num_classes=30):
        super(MyModel, self).__init__(name="baseWithPyramidPooling")

        # base model
        self.base_model = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)
        self.base_model.trainable = True

        # upsample (just temporary)
        self.up0 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")

        # pyramid pooling module
        # 
        # Stride = (input_size//output_size)
        # Kernel size = input_size - (output_size-1)*stride
        # Padding = 0

        self.features = []

        bins = [1,2,3,6]
        inp_dim = (128,256)
        for bin_ in bins:
          strides = (inp_dim[0] // bin_, inp_dim[1] // bin_)
          kernel_size = (inp_dim[0] - (bin_-1)*strides[0], inp_dim[1] - (bin_-1)*strides[1])
          self.features.append(tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(kernel_size, strides=strides),
            tf.keras.layers.Conv2D(512, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
          ]))

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
        
        poolings = [feature_map_upsampled]
        for f in self.features:
          poolings.append(tf.image.resize(f(feature_map_upsampled), [128, 256]))
        
        feature_map_bins_concat = self.concat0(poolings)
        print("F MAP " + str(feature_map_bins_concat))
        output_num_classes_depth = self.conv0(feature_map_bins_concat)
        output_upsampled = self.up1(output_num_classes_depth)
        output_softmax = self.soft0(output_upsampled)
        print("OUTPUT " + str(output_softmax))
        
        return output_softmax
      
model = MyModel(30)
train_ds = train_ds.batch(1)
val_ds = val_ds.batch(1)

model.compile(optimizer="SGD", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
#model.build(input_shape=(None,1024,2048,3))
model.fit(x=train_ds, validation_data=val_ds, epochs=3)                                 
model.summary(line_length=90)