import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

input_tensor = tf.constant([1,2,3,4,5])
dataset = tf.data.Dataset.from_tensor_slices(input_tensor)
dataset = dataset.make_one_shot_iterator()

for batch, num  in enumerate(dataset):
    print(batch, num)
    break

for batch, num  in enumerate(dataset):
    print(batch, num)
    break
