from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf
import yaml

import citycam_dataset_converter
import anchor_lib
import dataset_lib
import bbox_lib
import model_lib

tf.enable_eager_execution()

"""Build the dataset."""
tarfile_path = "/Users/Geoff/Documents/my_git/data/164.tar.gz"

train_filepath, val_filepath = citycam_dataset_converter.convert(tarfile_path)

output_h = 15
output_w = 22
input_shape_h = 240
input_shape_w = 352

anchor_strides = [input_shape_h / output_h, input_shape_w / output_w]

anchors = anchor_lib.anchor_gen(output_h, output_w, anchor_stride=anchor_strides)

batch_size = 32
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3
neg_label_value = -1
ignore_label_value = -2

dataset_builder_fn = functools.partial(
    dataset_lib.read_data,
    anchors=anchors,
    batch_size=batch_size,
    pos_iou_threshold=pos_iou_threshold,
    neg_iou_threshold=neg_iou_threshold,
    neg_label_value=neg_label_value,
    ignore_label_value=ignore_label_value)

epoch = 30
shuffle_buffer_size = 1000

train_ds = dataset_builder_fn(
    train_filepath, epoch=epoch,
    shuffle_buffer_size=shuffle_buffer_size,
    image_arg=True)

val_ds = dataset_builder_fn(val_filepath)
test_ds = dataset_builder_fn(val_filepath)


"""Build model."""
def build_model(num_classes, anchor_num_per_output):
    base_network_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights="imagenet")

    for layer in base_network_model.layers:
        layer.trainable = False

    h = base_network_model.get_layer(name="activation_39").output
    drop_rate = 0.5
    h = tf.keras.layers.Dropout(drop_rate)(h)

    classification_branch = tf.keras.layers.Conv2D(
        (num_classes + 1) * anchor_num_per_output, (1, 1))(
            h)
    regression_branch = tf.keras.layers.Conv2D(4 * anchor_num_per_output, (1, 1))(
        h)
    model_outputs = [classification_branch, regression_branch]
    return tf.keras.models.Model(base_network_model.input, model_outputs)


"""Training preperration. """
global_step = tf.train.get_or_create_global_step()

num_classes = 10
# len(anchor_scales) * len(anchor_aspect_ratio)
anchor_num_per_output = 9

od_model = build_model(num_classes, anchor_num_per_output)

# init for the learning.
learning_rate = 0.001
decay_step = 1000
decay_alpha = 0.000001

global_step = tf.train.get_or_create_global_step()
decayed_lr = tf.train.cosine_decay(
    learning_rate=learning_rate,
    global_step=global_step,
    decay_steps=decay_step,
    alpha=decay_alpha)

optimizer = tf.train.AdamOptimizer(decayed_lr)

# init for the loss.
classificaiton_loss_weight = 1
regression_loss_weight = 10
negative_ratio = 3

compute_loss_fn = functools.partial(
    model_lib.compute_loss,
    num_classes=num_classes,
    c_weight=classificaiton_loss_weight,
    r_weight=regression_loss_weight,
    neg_label_value=neg_label_value,
    ignore_label_value=ignore_label_value,
    negative_ratio=negative_ratio)

"""Training loop."""
val_iter = 100
val_batch = 5
test_iter = 500
test_batch = 10
score_threshold = 0.5
max_prediction = 100

train_loss_sum = 0
for train_index, train_item in enumerate(train_ds):
    with tf.GradientTape() as tape:
        train_network_output = od_model(train_item["image"], training=True)
        train_loss = compute_loss_fn(train_network_output,
                                           train_item["bboxes_preprocessed"],
                                           train_item["labels_preprocessed"])
        train_loss_sum += train_loss

        grads = tape.gradient(train_loss, od_model.variables)
        optimizer.apply_gradients(
            zip(grads, od_model.variables),
            global_step=tf.train.get_or_create_global_step())

    if train_index != 0 and train_index % val_iter == 0:
        val_loss_sum = 0
        for val_index, val_item in enumerate(val_ds):
            if val_index != 0 and val_index % val_batch == 0:
                break
            val_network_output = od_model(val_item["image"], training=False)
            val_loss = compute_loss_fn(val_network_output,
                                             val_item["bboxes_preprocessed"],
                                             val_item["labels_preprocessed"])
            val_loss_sum += val_loss

        train_loss = train_loss_sum / val_iter
        val_loss = val_loss_sum / val_batch

        print("Loss at step {:04d}: train loss: {:.3f}, val loss: {:3f}".format(
            train_index, train_loss, val_loss))

        train_loss_sum = 0


    if train_index != 0 and train_index % test_iter == 0:
        for test_index, test_item in enumerate(test_ds):
            if test_index != 0 and test_index % test_batch:
                break

            test_network_output = od_model(test_item["image"], training=False)

            bbox_list, label_list = model_lib.predict(
                test_network_output,
                mask=test_item["mask"],
                score_threshold=score_threshold,
                neg_label_value=neg_label_value,
                anchors=anchors,
                max_prediction=max_prediction,
                num_classes=num_classes)

            image_list = []
            for image, bbox, label in zip(test_item["image"], bbox_list, label_list):
                image += 0.5
                normalized_bboxes = bbox_lib.normalizing_bbox(
                    bbox, input_shape_h, input_shape_w)
                image_with_bboxes = tf.image.draw_bounding_boxes(
                    image[tf.newaxis, ...], normalized_bboxes[tf.newaxis, ...])
                image_list.append(image_with_bboxes)

    od_model.save_weights(".")
