# Requirements: pip3 install tensorflow_datasets

# The data is place under ~/tensorflow_datasets/.

# Following class loads in the dataset of scene parsing. There are methods to access training data and test data.
# Furthermore, you can get access to meta data also.
# Author: David Yu, Simon Sirak, Kristoffer Chammas
# Date: 2020-04-21
# Latest update: 2020-04-21
#
# Update: 2020-04-28
# Added function(s) for casting and normalizing the data

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self, dataset_name='scene_parse150', training_split='90%'):
        self.train_data, self.train_info = tfds.load(dataset_name, split='train[:' + training_split + "]", as_supervised=True, with_info=True)
        self.val_data, self.val_info = tfds.load(dataset_name, split='train[' + training_split + ":]", as_supervised=True, with_info=True)
        self.test_data, self.test_info = tfds.load(dataset_name, split='test', as_supervised=True, with_info=True)

    # function that uses map to cast and normalize data
    def normalizeAllData(self):
        self.train_data = self.train_data.map(preprocess_input)
        self.val_data = self.val_data.map(preprocess_input)
        self.test_data = self.test_data.map(preprocess_input)

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

# function that is used in the map function
# it casts and normalizes the data
# note that @tf.function has to be used to prevent unexpected exceptions
@tf.function
def preprocess_input(x, y):
    x = tf.dtypes.cast(x, tf.float32)
    return tf.keras.applications.resnet50.preprocess_input(x), y
