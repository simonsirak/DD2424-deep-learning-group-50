import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from load_data import DataLoader
from tensorflow.keras.models import model_from_json          

loader = DataLoader(dataset_name='cityscapes', split="20%")
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
        
        layer_3_conv2 = self.base_model.get_layer('conv4_block1_1_conv') # conv4_block1_1_conv (Conv2D)    (None, 64, 128, 256) 131328      conv3_block4_out[0][0]           
        layer_3_downsample = self.base_model.get_layer('conv4_block1_0_conv') # conv4_block1_0_conv (Conv2D)    (None, 64, 128, 1024 525312      conv3_block4_out[0][0]           

        layer_4_conv2 = self.base_model.get_layer('conv5_block1_1_conv') # conv5_block1_1_conv (Conv2D)    (None, 32, 64, 512)  524800      conv4_block6_out[0][0]           
        layer_4_downsample = self.base_model.get_layer('conv5_block1_0_conv') # conv5_block1_0_conv (Conv2D)    (None, 32, 64, 2048) 2099200     conv4_block6_out[0][0]           

        layer_3_conv2.strides = (1,1)
        layer_3_conv2.padding = 'same'
        layer_3_conv2.dilation_rate = (2,2)

        layer_3_downsample.strides = (1,1)

        layer_4_conv2.strides = (1,1)
        layer_4_conv2.padding = 'same'
        layer_4_conv2.dilation_rate = (4,4)

        layer_4_downsample.strides = (1,1)

        self.base_model = model_from_json(self.base_model.to_json())

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
        inp_dim = (64,128)
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
        #feature_map_upsampled = self.up0(feature_map)
        
        poolings = [feature_map]
        for f in self.features:
          poolings.append(tf.image.resize(f(feature_map), [64, 128]))
        
        feature_map_bins_concat = self.concat0(poolings)
        print("F MAP " + str(feature_map_bins_concat))
        output_num_classes_depth = self.conv0(feature_map_bins_concat)
        output_upsampled = self.up1(output_num_classes_depth)
        output_softmax = self.soft0(output_upsampled)
        print("OUTPUT " + str(output_softmax))
        
        return output_softmax
      
model = MyModel(30)
train_ds = train_ds.batch(4)
val_ds = val_ds.batch(4)
test_ds = test_ds.batch(4)

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=sgd, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"])                            

#model.build(input_shape=(None,1024,2048,3))
model.fit(x=train_ds, validation_data=val_ds, epochs=3)  
model.summary(line_length=90)
model.evaluate(test_ds)   