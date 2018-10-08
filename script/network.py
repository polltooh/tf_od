import functools
import tensorflow as tf

def inference(images, num_classes):
    h = tf.keras.layers.Conv2D(32, (3, 3), 2, 'valid', name='conv1')(images)
    print(h)
    with tf.name_scope("network"):
        h = tf.keras.layers.Conv2D(32, (3, 3), 2, 'valid')(images)
        h = tf.keras.layers.Conv2D(64, (3, 3), 2, 'valid')(h)
        h = tf.keras.layers.Conv2D(128, (3, 3), 2, 'valid')(h)
        h = tf.keras.layers.Conv2D(256, (3, 3), 2, 'valid')(h)
        with tf.name_scope("classification"):
            classification_branch = tf.keras.layers.Conv2D(num_classes, (1, 1), 1, 'valid', name='classification_branch')(h)

        with tf.name_scope("regression"):
            regression_branch = tf.keras.layers.Conv2D(4, (1, 1), 1, 'valid', name='regression_branch')(h)

    return [classification_branch, regression_branch]


def loss():
    classification_branch = functools.partial(tf.keras.backend.categorical_crossentropy, from_logits=True)
    regression_branch_loss = functools.partial(tf.losses.huber_loss, delta=1)
