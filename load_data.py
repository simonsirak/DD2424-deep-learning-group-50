# Requirements: pip3 install tensorflow_datasets

# The data is place under ~/tensorflow_datasets/.

# Following class loads in the dataset of scene parsing. There are methods to access training data and test data.
# Furthermore, you can get access to meta data also.
# Author: David Yu, Simon Sirak, Kristoffer Chammas
# Date: 2020-04-21
# Latest update: 2020-04-21

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class LoadData:
    def __init__(self):
        (self.data, self.data_info) = tfds.load('scene_parse150', with_info=True)

    def getAllData(self):
        return self.data

    def getTrainData(self):
        return self.data.get('train')

    def getTestData(self):
        return self.data.get('test')

    def getDataInfo(self):
        return self.data_info

def main():
    ld = LoadData()
    data = ld.getTestData()
    data_info = ld.getDataInfo()
    print("Main function")

    print(type(data))

if __name__ == '__main__':
    main()