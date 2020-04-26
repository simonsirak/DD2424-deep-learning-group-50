
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# nr of images
size = 50062
batch_size = 128

# Load dataset and split it how you want! You need to batch here if you use the Dataset API
# 45056
train_ds = tfds.load('cifar10', split='train[:90%]', as_supervised=True)
train_ds = train_ds.shuffle(1024)

print(train_ds)

# 5006
val_ds = tfds.load('cifar10', split='train[90%:]', as_supervised=True)
val_ds = val_ds.shuffle(1024)

# Create model class yourself if you want
class A1(tf.keras.Model):
  def __init__(self, lambda_, num_classes=10):
    super(A1, self).__init__(name="assignment_1")
    self.dense0 = tf.keras.layers.Flatten()
    self.bn0 = tf.keras.layers.BatchNormalization()
    self.dense1 = tf.keras.layers.Dense(50, activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=0.05), kernel_regularizer=tf.keras.regularizers.l2(lambda_))
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(50, activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=0.05), kernel_regularizer=tf.keras.regularizers.l2(lambda_))
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.dense3 = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=0.05), kernel_regularizer=tf.keras.regularizers.l2(lambda_))

  def call(self, inputs):
    inp = tf.dtypes.cast(inputs, tf.float32) / 255.0
    x = self.dense0(inp)
    x = self.bn0(x)
    x = self.dense1(x)
    x = self.bn1(x)
    x = self.dense2(x)
    x = self.bn2(x)
    #print(x)
    return self.dense3(x)

# tensorflow is trash and cannot work with SparseCategoricalCrossEntropy+MeanIoU, see this issue:
# https://github.com/tensorflow/tensorflow/issues/32875
class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
      # print("ACTUAL: " + str(y_true))
      # print("PREDICTED: " + str(y_pred))
      y_pred = tf.argmax(y_pred, axis=-1) # do this because I think the predicted value is always one-hot vector I THINK
      return super().__call__(y_true, y_pred, sample_weight=sample_weight)

num_classes = 10
model = A1(0.005, num_classes)

# TensorBoard and ModelCheckpoint callbacks would be awesome for visualization and saving models!
# Should plot loss and mIoU initially.

# compile and fit a given model to a given dataset, with given validation data as well
# should be able to plot measurements over time, e.g loss/cost and accuracy
# should save checkpoints every epoch that can be restored
# should be able to resume training from paused model
def train_model(model, train_dataset, val_dataset, num_classes, loss_fn, batch_size=64, epochs=10, backup_path=None):
  train_dataset = train_dataset.batch(batch_size)
  val_dataset = val_dataset.batch(batch_size)

  model.compile(optimizer="sgd",loss=loss_fn(), metrics=["accuracy", MeanIoU(num_classes=num_classes)])

  if(backup_path is not None):
    # Assume "epochs" has been adapted to train as long as is left at the point of this checkpoint.
    model.load_weights(backup_path)

  # y is part of x when x is a Dataset
  model.fit(x=train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[TensorBoard(), ModelCheckpoint("backup{epoch:02d}of" + str(epochs))])


##########################
# PLAYING AROUND/TESTING #
##########################

# Regular training of a model
# train_model(model, train_ds, val_ds, num_classes, SparseCategoricalCrossentropy, batch_size=batch_size, epochs=10)

# load test data for evaluation
# Either fitting or evaluation needs to be done before summary can be used, compiling is not enough!
# This is because custom models are defined through their call()-function, which is run when you use 
# the model.
test_ds = tfds.load('cifar10', split='test', as_supervised=True)
test_ds = test_ds.shuffle(1024)
test_ds = test_ds.batch(batch_size) # test data apparently needs to be batched with same size as training data

# load saved model (compiling needs to be done before loading weights for some reason, don't quite understand why)
halfway = A1(0.005, num_classes)

# model that resumed from halfway
train_model(halfway, train_ds, val_ds, num_classes, SparseCategoricalCrossentropy, batch_size=batch_size, epochs=5, backup_path="backup05of10")
res = halfway.evaluate(test_ds)
print()
print(" HALFWAY->FULL RESULTS ")
print(res)

# model that kept going from beginning to end, the halfway point came from the same training session as this model
full = A1(0.005, num_classes)
full.compile(optimizer="sgd",loss=SparseCategoricalCrossentropy(), metrics=["accuracy", MeanIoU(num_classes=num_classes)])
full.load_weights("backup10of10")
res = full.evaluate(test_ds)
print()
print(" BEGINNING->FULL RESULTS ")
print(res)

# The two above should yield similar results, +- randomness in training 