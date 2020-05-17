# Requirements: pip3 install tensorflow_datasets

# The data is place under ~/tensorflow_datasets/.

# Following class loads in the dataset of scene parsing. There are methods to access training data and test data.
# Furthermore, you can get access to meta data also.
# Author: David Yu, Simon Sirak, Kristoffer Chammas
# Date: 2020-04-21
# Latest update: 2020-04-21


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np 

#ignored labels (see the following for what labels are ignorde during evaluation: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
sample_weights = [0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1]

class DataLoader:
    def __init__(self, dataset_name='cityscapes', split='10%', batch_size=8):
        self.train_data, self.train_info = tfds.load(dataset_name, split='train[:' + split + ']', with_info=True)
        self.val_data, self.val_info = tfds.load(dataset_name, split='validation[:1]', with_info=True)
        self.test_data, self.test_info = tfds.load(dataset_name, split='test', with_info=True)
        self.batch_size = batch_size
        
    def prepareDatasets(self):
      #print(self.train_info)
      self.train_data = self.train_data.map(preprocess_input).cache().shuffle(512)
      self.train_data = self.train_data.map(rand_flip)
      self.train_data = self.train_data.batch(self.batch_size).prefetch(1) # batch before adding sample weights

      #print(self.val_info)
      self.val_data = self.val_data.map(preprocess_input).cache()
      self.val_data = self.val_data.batch(self.batch_size).prefetch(1) # batch before adding sample weights

      #print(self.test_info)
      self.test_data = self.test_data.map(preprocess_input).cache()
      self.test_data = self.test_data.batch(self.batch_size).prefetch(1) # batch before adding sample weights

    def getAllData(self):
        return (self.train_data, self.val_data, self.test_data)
    
    def getTrainData(self):
        return self.train_data
        
    def getValData(self):
        return self.val_data

    def getTestData(self):
        return self.test_data

    def getDataInfo(self):
        return (self.train_info, self.val_info, self.test_info)

@tf.function
def preprocess_input(x):
    xx = {}
    xx["image_left"] = x["image_left"]
    xx["segmentation_label"] = x["segmentation_label"]
    xx["image_left"] = tf.dtypes.cast(xx["image_left"], tf.float32)
    xx["image_left"] = tf.keras.applications.resnet.preprocess_input(xx["image_left"])
    
    xx["image_left"] = tf.image.resize(xx["image_left"], (256,512), method='nearest')
    xx["segmentation_label"] = tf.image.resize(xx["segmentation_label"], (256,512), method='nearest')

    return (xx["image_left"], xx["segmentation_label"])

# done before batching
@tf.function
def rand_flip(x, y):
  image = x
  #tf.print(image.shape)
  label = y
  #tf.print(sample_weights.shape)
  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
  return (image, label)