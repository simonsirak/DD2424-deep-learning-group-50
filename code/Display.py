import tensorflow as tf
from IPython.display import clear_output
#mpl.use('Agg')
import matplotlib.pyplot as plt

def display(display_list, name='before'):
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

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  display([sample_image[0], sample_mask[0],
            create_mask(model.predict(sample_image))],'after')


class DisplayCallback(tf.keras.callbacks.Callback):
  # def __init__(self, model):
  #   super(DisplayCallback, self).__init__()
  #
  #   self.model = model


  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
