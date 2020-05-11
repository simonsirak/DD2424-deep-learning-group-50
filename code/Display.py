import tensorflow as tf
from IPython.display import clear_output
#mpl.use('Agg')
import matplotlib.pyplot as plt

class DisplayCallback(tf.keras.callbacks.Callback):
  def __init__(self, model, dataset):
    super(DisplayCallback, self).__init__()
  
    self.model = model

    self.dataset = dataset
    

  def display(self, display_list, name='before'):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
      plt.axis('off')
    plt.show()

  # for image, mask in train_ds.take(1):
  #   sample_image, sample_mask = image, mask
  # display([sample_image[0], sample_mask[0]], 'before')

  def create_mask(self, pred_mask):
    print(tf.math.reduce_max(pred_mask, axis=-1))
    pred_mask = tf.argmax(pred_mask, axis=-1)
    print(pred_mask)
    pred_mask = tf.dtypes.cast(tf.expand_dims(pred_mask,-1), tf.uint64)
    #print(pred_mask.shape)
    #print(pred_mask)
    return pred_mask[0]

  def show_predictions(self, dataset=None, num=1):
    for image, mask in self.dataset.take(1):
      self.model.evaluate(self.dataset)
      pred_mask = self.model(image, training=False)
      
      # print(image)
      # print(pred_mask.shape)
      # print(mask.shape)
      self.display([image[0], mask[0], self.create_mask(pred_mask)])

  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    if epoch % 2 == 0:
      self.show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
