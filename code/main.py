
# Authors: David Yu, Kristoffer Chammas, Simon Sirak
# Date: 2020-04-21
# Latest update: 2020-05-17

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import model_from_json          

from display import DisplayCallback
from load_data import DataLoader
from model import DilatedResNet, PSPNet

# Tensorflow 2.1 cannot work with SparseCategoricalCrossEntropy+MeanIoU, see this issue: https://github.com/tensorflow/tensorflow/issues/32875
# As such, we had to implement the following workaround for the MeanIoU.
class MeanIoU(tf.keras.metrics.MeanIoU):
  def __call__(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.argmax(y_pred, axis=-1) # The predicted value is a one-hot vector, so this needs to be converted
    y_pred = tf.dtypes.cast(tf.expand_dims(y_pred,-1), tf.uint8)

    return super().__call__(y_true, y_pred, sample_weight=sample_weight)

def train_model(model, train_dataset, val_dataset, num_classes, loss_fn, epochs=10):
  sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  model.compile(optimizer=sgd,loss=loss_fn(), weighted_metrics=["accuracy", MeanIoU(num_classes)])

  # verbose=2 gives 1 line per epcoh, which is good when logging to a file
  model.fit(x=train_dataset, verbose=1, epochs=epochs, validation_data=val_dataset, callbacks=[TensorBoard(), ModelCheckpoint("backup" + str(epochs)), DisplayCallback(model, train_dataset, saveimg=True)])

# Create dataset loader

# NOTE: A simple overfit-test can be made by loading only 1 image, but the generated 
# images will look bad. This is due to poor estimation of batch norm parameters, and 
# can be hotfixed by switching to training mode during image generation in display.py
loader = DataLoader('cityscapes', '50%', batch_size=8)

# Prepare all the data before usage. This includes:
# - Casting the images to float32.
# - Normalizing all the data according to the normalization strategy for ResNet50. 
# - Applying a random flip to the training data every time it is encountered.
# - Batching the data.
# - Prefetching 1 batch to improve the data throughput.

loader.prepareDatasets()

# Load the datasets
train_ds = loader.getTrainData()
val_ds = loader.getValData()

# Define model
num_classes = 34
model = PSPNet(feature_dim=(256,512,2048), num_classes=num_classes, use_ppm=True, bins=[1, 2, 3, 6])

# Train the model
train_model(model, train_ds, val_ds, num_classes, SparseCategoricalCrossentropy, epochs=60)