# Requirements: pip3 install tensorflow_datasets

# The data is place under ~/tensorflow_datasets/.

# Following class loads in the dataset of scene parsing. There are methods to access training data and test data.
# Furthermore, you can get access to meta data also.
# Author: David Yu, Simon Sirak, Kristoffer Chammas
# Date: 2020-04-21
# Latest update: 2020-04-21


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

class DataLoader:
    def __init__(self, dataset_name='oxford_iiit_pet', split='10%'):
        self.train_data, self.train_info = tfds.load(dataset_name, split='train[:' + split + ']', with_info=True)
        self.test_data, self.test_info = tfds.load(dataset_name, split='test', with_info=True)
        
    def normalizeAllData(self):
        self.train_data = self.train_data.map(preprocess_input)
        self.test_data = self.test_data.map(preprocess_input)
        
    def getAllData(self):
        return (self.train_data, self.test_data)
    
    def getTrainData(self):
        return self.train_data

    def getTestData(self):
        return self.test_data

    def getDataInfo(self):
        return (self.train_info, self.test_info)

@tf.function
def preprocess_input(x):
    xx = {}
    xx["image"] = x["image"]
    xx["segmentation_mask"] = x["segmentation_mask"]
    xx["image"] = tf.dtypes.cast(xx["image"], tf.float32)
    xx["image"] = tf.keras.applications.resnet.preprocess_input(xx["image"])
    xx["image"] = tf.image.resize_with_pad(xx["image"], 512, 512, method="bilinear")
    xx["segmentation_mask"] = tf.image.resize_with_pad(xx["segmentation_mask"], 512, 512, method="nearest")
    return (xx["image"], xx["segmentation_mask"])
