import functools
import tensorflow as tf

tf.enable_eager_execution()


class ObjectDetectionModel(tf.keras.Model):
  def __init__(self, class_num):
    super(ObjectDetectionModel, self).__init__()
    self.class_num = class_num
    self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), 2)
    self.c_conv = tf.keras.layers.Conv2D(class_num, (1, 1), 1)
    self.r_conv = tf.keras.layers.Conv2D(class_num, (1, 1), 1)

  def call(self, inputs, training):
    """Run the model."""
    result = self.conv1(inputs)
    c_out = self.c_conv(result)
    c_out = tf.keras.layers.Reshape((-1, 10), name='c_out')(c_out)
    r_out = self.r_conv(result)
    # r_out = tf.keras.layers.Reshape((-1, 4), name='l_out')(r_out)
    return c_out, r_out

def loss(model, inputs, targets):
  error = model(inputs)
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)


class_num = 10
anchor_num = 12321

image_input = tf.ones([1, 240, 352, 3], tf.float32)
image_input = tf.random_normal([1, 240, 352, 3])

model = ObjectDetectionModel(class_num)
target = None
grad(model, image_input, target)
c_out, r_out = model(image_input, training=True)

# huber_loss = functools.partial(tf.losses.huber_loss, 
c_loss = functools.partial(tf.keras.backend.categorical_crossentropy, from_logits=True)
l_loss = tf.losses.huber_loss

optimizer = tf.keras.optimizers.Adam(lr=0.1)

model.compile(optimizer=optimizer,
              loss={'c_out': c_loss, 'l_out': l_loss},
              loss_weights={'c_out': 1., 'l_out': 0.2})


image_data = tf.ones((1, 224, 224, 3), tf.float32)
label_data = tf.ones((1, anchor_num), tf.int32)
label_data_one_hot = tf.one_hot(label_data, class_num)
bbox_data = tf.ones((1, anchor_num, 4), tf.float32)

model.fit({'image': image_data},
          {'c_out': label_data_one_hot, 'l_out': bbox_data},
          steps_per_epoch=50)
