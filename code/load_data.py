# Authors: David Yu, Kristoffer Chammas, Simon Sirak
# Date: 2020-04-21
# Latest update: 2020-05-17

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np 

class DataLoader:
  def __init__(self, dataset_name='cityscapes', split='10%', batch_size=8):
    self.train_data, self.train_info = tfds.load(dataset_name, split='train[:' + split + ']', with_info=True)
    self.val_data, self.val_info = tfds.load(dataset_name, split='validation', with_info=True)
    self.test_data, self.test_info = tfds.load(dataset_name, split='test', with_info=True)
    self.batch_size = batch_size
      
  def prepareDatasets(self):
    self.train_data = self.train_data.map(preprocess_input).cache().shuffle(512)
    self.train_data = self.train_data.map(rand_flip)
    self.train_data = self.train_data.batch(self.batch_size).prefetch(1)

    self.val_data = self.val_data.map(preprocess_input).cache()
    self.val_data = self.val_data.batch(self.batch_size).prefetch(1)

    self.test_data = self.test_data.map(preprocess_input).cache()
    self.test_data = self.test_data.batch(self.batch_size).prefetch(1)

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
  label = y
  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
  return (image, label)