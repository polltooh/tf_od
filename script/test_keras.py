import functools
import tensorflow as tf

class_num = 10
anchor_num = 12321

image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image')

# bbox = tf.convert_to_tensor([[10, 10, 30, 30], [40, 40, 60, 60]], tf.float32)
bbox = tf.keras.layers.Input((anchor_num, 4), name='bbox')
# label = tf.convert_to_tensor([[1], [1]], tf.float32)
label = tf.keras.layers.Input((anchor_num, 1), name='label')

x = tf.keras.layers.Conv2D(64, (3, 3), 2)(image_input)
c_out = tf.keras.layers.Conv2D(class_num, (1, 1), 1)(x)
c_out = tf.keras.layers.Reshape((-1, 10), name='c_out')(c_out)

l_out = tf.keras.layers.Conv2D(4, (1, 1), 1)(x)
l_out = tf.keras.layers.Reshape((-1, 4), name='l_out')(l_out)

model = tf.keras.Model(inputs=[image_input], outputs=[c_out, l_out])

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
