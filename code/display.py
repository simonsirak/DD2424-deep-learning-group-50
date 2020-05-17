# Authors: David Yu, Kristoffer Chammas, Simon Sirak
# Date: 2020-04-21
# Latest update: 2020-05-17

import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib as mpl

class DisplayCallback(tf.keras.callbacks.Callback):
  """A callback that generates images every 5:th epoch.
  
  If saveimg=True, then the image will be directly output
  to a .png file. Otherwise, the image is presented in an
  interactive plot, with the option to save it from there.
  """
  def __init__(self, model, dataset, saveimg):
    super(DisplayCallback, self).__init__()
  
    self.model = model
    self.dataset = dataset
    self.saveimg = saveimg

    if(saveimg):
      mpl.use('Agg')

  def display(self, display_list, name='before', epoch=None):
    fig = plt.figure()
    if(epoch is not None):
      fig.suptitle("Epoch: " + str(epoch))

    for i in range(len(display_list)):
      (title, img) = display_list[i]
      plt.subplot(1, len(display_list), i+1)
      plt.title(title)
      plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
      plt.axis('off')

    if(self.saveimg):
      plt.savefig(name)
    else:
      plt.show()

  def create_mask(self, pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.dtypes.cast(tf.expand_dims(pred_mask,-1), tf.uint8)
    return pred_mask[0]

  def show_predictions(self, training=False, name='result', epoch=None):
    for image, mask in self.dataset.take(1):
      pred_mask = self.model(image, training=training)

      # - if epoch is None, output all three
      # - the first epoch should only show input image and true mask
      # - otherwise, show true and predicted
      if(epoch is None):
        self.display([('Input Image',image[0]), ('True Mask', mask[0]), ('Predicted Mask', self.create_mask(pred_mask))], name=name)
      elif(epoch == 1):
        self.display([('Input Image',image[0]), ('True Mask', mask[0])], name=name, epoch=epoch)
      else:
        self.display([('True Mask', mask[0]), ('Predicted Mask', self.create_mask(pred_mask))], name=name, epoch=epoch)

  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    
    # produce images initially and every 5th epoch
    if(epoch == 0 or (epoch+1) % 5 == 0):
      # NOTE: Set training=True if you intend to do some kind of overfitting-test, 
      # since these tests do not have enough data/batches to compute good batch 
      # normalization means and standard deviations.
      self.show_predictions(training=False, name="img_" + str(epoch+1), epoch=epoch+1)