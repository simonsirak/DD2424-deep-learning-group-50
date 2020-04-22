
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU

# nr of images
size = 50062
batch_size = 64

# Load dataset and split it how you want! You need to batch here if you use the Dataset API
# 45056
train_ds = tfds.load('cifar10', split='train[:90%]', as_supervised=True)
train_ds = train_ds.shuffle(1024)
train_ds = train_ds.batch(batch_size)

# 5006
val_ds = tfds.load('cifar10', split='train[90%:]', as_supervised=True)
val_ds = val_ds.shuffle(1024)
val_ds = val_ds.batch(batch_size)

# Create model class yourself if you want
class A1(tf.keras.Model):
  def __init__(self, lambda_, num_classes=10):
    super(A1, self).__init__(name="assignment_1")
    self.dense0 = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(50, activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=0.05), kernel_regularizer=tf.keras.regularizers.l2(lambda_))
    self.dense2 = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=0.05), kernel_regularizer=tf.keras.regularizers.l2(lambda_))

  def call(self, inputs):
    inp = tf.dtypes.cast(inputs, tf.float32) / 255.0
    x = self.dense0(inp)
    x = self.dense1(x)
    x = self.dense2(x)
    print(x)
    return x

model = A1(0.005, 10)



model.compile(optimizer="sgd",loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])#MeanIoU(num_classes=10)])

# y is part of x when x is a Dataset
model.fit(x=train_ds, epochs=10, validation_data=val_ds, callbacks=[TensorBoard(), ModelCheckpoint("backup")])

# TensorBoard and ModelCheckpoint callbacks would be awesome for visualization and saving models!
# Should plot loss and mIoU initially.

# compile and fit a given model to a given dataset, with given validation data as well
# should be able to plot measurements over time, e.g loss/cost and accuracy
# should save checkpoints every epoch that can be restored
# should be able to resume training from paused model
def train_model(model, train_dataset, val_dataset, num_classes, loss_fn, batch_size=64, epochs=10):
  train_dataset = train_dataset.batch(batch_size)
  val_dataset = val_dataset.batch(batch_size)

  model.compile(optimizer="sgd",loss=loss_fn(), metrics=["accuracy", MeanIoU(num_classes=num_classes)])

  # y is part of x when x is a Dataset
  model.fit(x=train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[TensorBoard(), ModelCheckpoint("backup")])

#train_model(model, train_ds, val_ds, 10, SparseCategoricalCrossentropy, batch_size=batch_size, epochs=10)