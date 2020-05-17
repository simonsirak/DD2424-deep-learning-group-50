# Authors: David Yu, Kristoffer Chammas, Simon Sirak
# Date: 2020-04-21
# Latest update: 2020-05-17

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from load_data import DataLoader
from tensorflow.keras.models import model_from_json          

class DilatedResNet(tf.keras.Model):
  """A ResNet50 model with a dilation strategy that reduces the output stride to 8.
  
  Dilation is applied to the last two blocks of the ResNet50 that have stride > 1. 
  The stride is reduced to 1 and dilation is used to get a bigger receptive field.
  """
  def __init__(self):
    super(DilatedResNet, self).__init__(name="DilatedResNet")

    self.dilated_resnet = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)
    
    # Each of the reductions are applied to both the residual path and the skip path

    layer_3_conv2 = self.dilated_resnet.get_layer('conv4_block1_1_conv') # residual path
    layer_3_downsample = self.dilated_resnet.get_layer('conv4_block1_0_conv') # skip path

    layer_4_conv2 = self.dilated_resnet.get_layer('conv5_block1_1_conv') 
    layer_4_downsample = self.dilated_resnet.get_layer('conv5_block1_0_conv')          

    layer_3_conv2.strides = (1,1)
    layer_3_conv2.padding = 'same'
    layer_3_conv2.dilation_rate = (2,2)

    layer_3_downsample.strides = (1,1)

    layer_4_conv2.strides = (1,1)
    layer_4_conv2.padding = 'same'
    layer_4_conv2.dilation_rate = (4,4)

    layer_4_downsample.strides = (1,1)

    # Simple way to apply the changes made above
    self.dilated_resnet = model_from_json(self.dilated_resnet.to_json())
    self.dilated_resnet.trainable = True
    self.output_stride = 8 # this is helpful in the PSPNet

  def call(self, input, training=False):
    return self.dilated_resnet(input,training=training)

class PSPNet(tf.keras.Model):
  """A PSPNet implementation.
  
  The number of classes can be specified in case another dataset is used.
  
  Omitting the Pyramid Pooling Module (PPM) will give the baseline model
  used during experiments. 
  """
  def __init__(self, feature_dim, num_classes=34, use_ppm=False, bins=[1, 2, 3, 6]):
    super(PSPNet, self).__init__(name="PSPNet")
    assert len(bins) == 4, "Length of bins should be equals 4"
    
    # Base model
    self.base_model = DilatedResNet()

    # Decide whether to use PPM or not
    self.use_ppm = use_ppm 

    # BEGIN Pyramid Pooling Module (PPM)

    if use_ppm:
      self.features = []

      bins = bins

      # Calculate stride and kernel size to get desired output size
      self.reduced_dim = [feature_dim[0] // self.base_model.output_stride, feature_dim[1] // self.base_model.output_stride]
      for bin_ in bins:
        strides = (self.reduced_dim[0] // bin_, self.reduced_dim[1] // bin_)
        kernel_size = (self.reduced_dim[0] - (bin_-1)*strides[0], self.reduced_dim[1] - (bin_-1)*strides[1])
        self.features.append(tf.keras.Sequential([
          tf.keras.layers.MaxPooling2D(kernel_size, strides=strides),
          tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
        ]))

      # concatenation layer
      self.concat0 = tf.keras.layers.Concatenate()

    # END Pyramid Pooling Module (PPM)

    # PPM produces a concatenated feature map with twice the amount 
    # of channels as the feature map produced by the baseline.

    feature_map_channels = feature_dim[2]

    # Final layers to produce output
    self.conv0 = tf.keras.layers.Conv2D(filters=2*feature_map_channels, kernel_size=(3, 3), padding='same', use_bias=False)
    self.bn0 = tf.keras.layers.BatchNormalization()
    self.relu0 = tf.keras.layers.ReLU()
    self.do = tf.keras.layers.Dropout(rate=0.1)
    self.conv1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1))
    self.up1 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear") # upsample with factor 8 to restore original input size
    self.soft0 = tf.keras.layers.Softmax()

  def call(self, input, training=False):
      # it is important that "training" is passed to any batch normalization or dropout layers, 
      # as they operate in different modes during training and testing/evaluation.

      x = self.base_model(input, training=training)

      if self.use_ppm:
        poolings = [x]
        for f in self.features:
          poolings.append(tf.image.resize(f(x, training=training), self.reduced_dim))
      
        x = self.concat0(poolings)

      x = self.conv0(x) 
      x = self.bn0(x,training=training)
      x = self.relu0(x)
      x = self.do(x,training=training)
      x = self.conv1(x)

      output_upsampled = self.up1(x)
      
      return self.soft0(output_upsampled)