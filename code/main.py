
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import model_from_json          
from Display import DisplayCallback
from load_data import DataLoader

# nr of images
size = 50062
batch_size = 1

# Load dataset and split it how you want! You need to batch here if you use the Dataset API
# 45056

loader = DataLoader('oxford_iiit_pet', '1')

# normalize all the data before usage
# this includes casting the images to float32
# as well as using the preprocess_input function from tf.keras.applications.resnet50
# (removed casting and normalizing from the call function)
loader.normalizeAllData()

train_ds = loader.getTrainData()
#train_ds = train_ds.shuffle(16)

# 5006
test_ds = loader.getTestData()
#test_ds = test_ds.shuffle(1024)

class DilatedResNet(tf.keras.Model):
  def __init__(self):
    super(DilatedResNet, self).__init__(name="DilatedResNet")

    self.dilated_resnet = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)
    
    layer_3_conv2 = self.dilated_resnet.get_layer('conv4_block1_1_conv') # conv4_block1_1_conv (Conv2D)    (None, 64, 128, 256) 131328      conv3_block4_out[0][0]           
    layer_3_downsample = self.dilated_resnet.get_layer('conv4_block1_0_conv') # conv4_block1_0_conv (Conv2D)    (None, 64, 128, 1024 525312      conv3_block4_out[0][0]           

    layer_4_conv2 = self.dilated_resnet.get_layer('conv5_block1_1_conv') # conv5_block1_1_conv (Conv2D)    (None, 32, 64, 512)  524800      conv4_block6_out[0][0]           
    layer_4_downsample = self.dilated_resnet.get_layer('conv5_block1_0_conv') # conv5_block1_0_conv (Conv2D)    (None, 32, 64, 2048) 2099200     conv4_block6_out[0][0]           

    layer_3_conv2.strides = (1,1)
    layer_3_conv2.padding = 'same'
    layer_3_conv2.dilation_rate = (2,2)

    layer_3_downsample.strides = (1,1)

    layer_4_conv2.strides = (1,1)
    layer_4_conv2.padding = 'same'
    layer_4_conv2.dilation_rate = (4,4)

    layer_4_downsample.strides = (1,1)

    self.dilated_resnet = model_from_json(self.dilated_resnet.to_json())
    self.dilated_resnet.trainable = True
    self.output_stride = 8

  def call(self, input, training=False):
    return self.dilated_resnet(input,training=training)

class PSPNet(tf.keras.Model):

  def __init__(self, inp_dim, num_classes=34, use_ppm=False, bins=[1, 2, 3, 6]):
    super(PSPNet, self).__init__(name="PSPNet")
    assert len(bins) == 4, "Length of bins should be equals 4"
    # base model
    self.base_model = DilatedResNet()
    self.use_ppm = use_ppm # Decide whether to use PPM or not

    # pyramid pooling module

    # Stride = (input_size//output_size)
    # Kernel size = input_size - (output_size-1)*stride
    # Padding = 0
    if use_ppm:
      self.features = []

      bins = bins
      self.reduced_dim = [inp_dim[0] // self.base_model.output_stride, inp_dim[1] // self.base_model.output_stride]
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

    inp_feature_maps = inp_dim[2]
    # final convolution layer
    # #features = 2*#inp_feature_maps because the 4 concatenated pooled maps add up to #inp_feature_maps in depth
    self.conv0 = tf.keras.layers.Conv2D(filters=2*inp_feature_maps, kernel_size=(3, 3), padding='same', use_bias=False)
    self.bn0 = tf.keras.layers.BatchNormalization()
    self.relu0 = tf.keras.layers.ReLU()
    self.conv1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1))

    # upsample layer (temporary)
    self.up1 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")

    # softmax
    self.soft0 = tf.keras.layers.Softmax()

  def call(self, input, training=False):
      # feature_map
      x = self.base_model(input, training=training)

      if self.use_ppm:
        poolings = [x] # feature_map
        for f in self.features:
          poolings.append(tf.image.resize(f(x, training=training), self.reduced_dim)) # feature_map
      
        x = self.concat0(poolings) # feature_map_bins_concat

      # print("F MAP " + str(feature_map_bins_concat))
      x = self.conv0(x) # output_num_classes_depth | feature_map_bins_concat
      x = self.bn0(x,training=training) # output_num_classes_depth
      x = self.relu0(x)
      x = self.conv1(x)
      output_upsampled = self.up1(x)

      # print("OUTPUT " + str(output_softmax))
      
      return self.soft0(output_upsampled)
      
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

# tensorflow is trash and cannot work with SparseCategoricalCrossEntropy+MeanIoU, see this issue:
# https://github.com/tensorflow/tensorflow/issues/32875
class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
      #print("ACTUAL: " + str(y_true))
      y_pred = tf.argmax(y_pred, axis=-1) # do this because I think the predicted value is always one-hot vector I THINK
      y_pred = tf.dtypes.cast(tf.expand_dims(y_pred,-1), tf.uint8)
      #print("PREDICTED: " + str(y_pred))

      return super().__call__(y_true, y_pred, sample_weight=sample_weight)


num_classes = 4
model = PSPNet(inp_dim=(512,512,2048), num_classes=num_classes, use_ppm=True, bins=[1, 2, 3, 6])

# TensorBoard and ModelCheckpoint callbacks would be awesome for visualization and saving models!
# Should plot loss and mIoU initially.

# compile and fit a given model to a given dataset, with given validation data as well
# should be able to plot measurements over time, e.g loss/cost and accuracy
# should save checkpoints every epoch that can be restored
# should be able to resume training from paused model
def train_model(model, train_dataset, val_dataset, num_classes, loss_fn, batch_size=64, epochs=10, backup_path=None):
  sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  model.compile(optimizer=sgd,loss=loss_fn(), metrics=["accuracy", MeanIoU(num_classes)])

  if(backup_path is not None):
    # Assume "epochs" has been adapted to train as long as is left at the point of this checkpoint.
    model.load_weights(backup_path)

  # y is part of x when x is a Dataset
  model.fit(x=train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[DisplayCallback(model, train_ds)])


##########################
# PLAYING AROUND/TESTING #
##########################

# Regular training of a model
train_model(model, train_ds, test_ds, num_classes, SparseCategoricalCrossentropy, batch_size=batch_size, epochs=2)

# cb.show_predictions(dataset=train_ds, num=1)
print(model.predict(train_ds))

# load test data for evaluation
# Either fitting or evaluation needs to be done before summary can be used, compiling is not enough!
# This is because custom models are defined through their call()-function, which is run when you use 
# the model.
# test_ds = loader.getTestData()
# test_ds = test_ds.shuffle(1024)
# test_ds = test_ds.batch(batch_size) # test data apparently needs to be batched with same size as training data

# # load saved model (compiling needs to be done before loading weights for some reason, don't quite understand why)
# halfway = DummyModel(0.005, num_classes)

# # model that resumed from halfway
# train_model(halfway, train_ds, val_ds, num_classes, SparseCategoricalCrossentropy, batch_size=batch_size, epochs=5, backup_path="backup05of10")
# res = halfway.evaluate(test_ds)
# print()
# print(" HALFWAY->FULL RESULTS ")
# print(res)

# # model that kept going from beginning to end, the halfway point came from the same training session as this model
# full = DummyModel(0.005, num_classes)
# full.compile(optimizer="sgd",loss=SparseCategoricalCrossentropy(), metrics=["accuracy", MeanIoU(num_classes=num_classes)])
# full.load_weights("backup10of10")
# res = full.evaluate(test_ds)
# print()
# print(" BEGINNING->FULL RESULTS ")
# print(res)

# The two above should yield similar results, +- randomness in training 
# model = PSPNet((512,1024),num_classes)
# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# model.compile(optimizer=sgd,loss=SparseCategoricalCrossentropy(), metrics=["accuracy", MeanIoU(num_classes)])
# model.load_weights("backup03of10")
# show_predictions()
