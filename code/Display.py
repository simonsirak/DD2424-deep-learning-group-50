import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib as mpl

class DisplayCallback(tf.keras.callbacks.Callback):
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

  # for image, mask in train_ds.take(1):
  #   sample_image, sample_mask = image, mask
  # display([sample_image[0], sample_mask[0]], 'before')

  def create_mask(self, pred_mask):
    #print(tf.math.reduce_max(pred_mask, axis=-1))
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.dtypes.cast(tf.expand_dims(pred_mask,-1), tf.uint8)
    #print(pred_mask.shape)
    #print(pred_mask)
    # print("PRED: " + str(pred_mask[0]))
    return pred_mask[0]

  def show_predictions(self, dataset=None, num=1, training=False, name='result', epoch=-1):
    for image, mask in self.dataset.take(1):
      pred_mask = self.model(image, training=training)
      # print("REAL: " + str(mask))
      # print(image)
      # print(pred_mask.shape)
      # print(mask.shape)

      # - if epoch == -1, output all three
      # - the first epoch should only show input image and true mask
      # - otherwise, show true and predicted
      if(epoch == -1):
        self.display([('Input Image',image[0]), ('True Mask', mask[0]), ('Predicted Mask', self.create_mask(pred_mask))], name=name)
      elif(epoch == 1):
        self.display([('Input Image',image[0]), ('True Mask', mask[0])], name=name, epoch=epoch)
      else:
        self.display([('True Mask', mask[0]), ('Predicted Mask', self.create_mask(pred_mask))], name=name, epoch=epoch)

  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    
    # show prediction every 5th epoch, first image has only image and label,
    # all remaining have label and prediction
    if(epoch == 0 or (epoch+1) % 5 == 0):
      # NOTE: Set training=True if you intend to do some kind of overfitting-test, 
      # since these tests do not have enough data/batches to compute good batch 
      # normalization means and standard deviations.
      self.show_predictions(training=False, name="img_" + str(epoch+1), epoch=epoch+1)

    # print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
