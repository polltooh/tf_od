from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow as tf
import yaml

import anchor_lib
import dataset_builder
import bbox_lib
import model_builder
import utils

tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def normalizing_bboxes(bboxes, image_height, image_width):
    if tf.rank(bboxes).numpy() == 2:
        axis = 1
    elif tf.rank(bboxes).numpy() == 3:
        axis = 2
    else:
        raise Exception(
            "bboxes dimention has to be 2 or 3, but get {} instead".format(tf.rank(bboxes)))

    y_min, x_min, y_max, x_max = tf.split(
        value=bboxes, num_or_size_splits=4, axis=axis)

    normalized_bboxes = tf.concat(
        [y_min / image_height, x_min / image_width,
            y_max / image_height, x_max / image_width],
        axis=axis)
    return normalized_bboxes


def add_item_summary(item, network_output, writer, anchors, num_classes):
    with writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device("/cpu:0"):
        image = item["image"]
        bboxes = item["bboxes"]
        batch_size, image_height, image_width, _ = image.get_shape().as_list()

        normalized_bboxes = normalizing_bboxes(
            bboxes, image_height, image_width)
        image_with_bboxes = tf.image.draw_bounding_boxes(
            tf.image.convert_image_dtype(image, tf.float32), normalized_bboxes)
        tf.contrib.summary.image('image_with_bboxes', image_with_bboxes)

        bboxes_preprocessed = item["bboxes_preprocessed"]
        convert_fn = functools.partial(
            bbox_lib.decode_box_with_anchor, anchors=anchors)
        bboxes_decoded = tf.map_fn(convert_fn, bboxes_preprocessed)
        bboxes_decoded_norm = normalizing_bboxes(
            bboxes_decoded, image_height, image_width)
        image_with_bboxes_converted = tf.image.draw_bounding_boxes(
            tf.image.convert_image_dtype(image, tf.float32), bboxes_decoded_norm)
        tf.contrib.summary.image(
            'image_with_bboxes_converted', image_with_bboxes_converted)

        labels_preprocessed = item["labels_preprocessed"]
        labels_preprocessed = tf.reshape(
            labels_preprocessed, [batch_size, 14, 21, -1])
        labels_heatmap = tf.reduce_any(tf.logical_and(tf.not_equal(
            labels_preprocessed, -2), tf.not_equal(labels_preprocessed, -1)), -1, keepdims=True)
        labels_heatmap = tf.cast(labels_heatmap, tf.float32)
        tf.contrib.summary.image('labels_heatmap', labels_heatmap)

        predict_heatmap = network_output["classification_output"]
        anchor_num_per_output = int(
            predict_heatmap.get_shape().as_list()[-1] / num_classes)

        for i in range(anchor_num_per_output):
            # first index is the background.
            current_predict_heatmap = tf.nn.softmax(
                predict_heatmap[..., i * (num_classes + 1): (i + 1) * (num_classes + 1)], -1)
            current_predict_heatmap = tf.reduce_max(
                current_predict_heatmap[..., 1:], -1)
            current_predict_heatmap = tf.cast(
                current_predict_heatmap * 255, tf.uint8)
            tf.contrib.summary.image('predict_heatmap_{}'.format(
                i), current_predict_heatmap[..., tf.newaxis])


def add_scalar_summary(value_list, name_list, writer):
    with writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device("/cpu:0"):
        for value, name in zip(value_list, name_list):
            tf.contrib.summary.scalar(name, value)


if __name__ == "__main__":
    with open("config.yaml", 'r') as config_file:
        config = yaml.load(config_file)

    global_step = tf.train.get_or_create_global_step()
    train_summary_writer = tf.contrib.summary.create_file_writer(
        os.path.join(config["summary"]["summary_dir"], "train"),
        flush_millis=config["summary"]["flush_millis"])

    val_summary_writer = tf.contrib.summary.create_file_writer(
        os.path.join(config["summary"]["summary_dir"], "val"),
        flush_millis=config["summary"]["flush_millis"])

    test_summary_writer = tf.contrib.summary.create_file_writer(
        os.path.join(config["summary"]["summary_dir"], "test"),
        flush_millis=config["summary"]["flush_millis"])

    get_output_shape_fn = functools.partial(
        utils.get_output_shape, kernel_size=config["network"]["kernel_size"],
        strides=config["network"]["strides"],
        layer_repeat_num=config["network"]["layer_repeat_num"])

    output_h = get_output_shape_fn(config["dataset"]["input_shape_h"])
    output_w = get_output_shape_fn(config["dataset"]["input_shape_w"])
    anchor_strides = [config["dataset"]["input_shape_h"] / output_h,
                      config["dataset"]["input_shape_w"] / output_w]

    anchors = anchor_lib.anchor_gen(
        output_h, output_w, scales=config["anchor"]["scales"],
        aspect_ratios=config["anchor"]["aspect_ratio"],
        base_anchor_size=config["anchor"]["base_anchor_size"],
        anchor_stride=anchor_strides)

    dataset_builder_fn = functools.partial(
        dataset_builder.read_data, anchors=anchors,
        batch_size=config["train"]["batch_size"],
        pos_iou_threshold=config["train"]["pos_iou_threshold"],
        neg_iou_threshold=config["train"][
            "neg_iou_threshold"], neg_label_value=config["dataset"]["neg_label_value"],
        ignore_label_value=config["dataset"]["ignore_label_value"])

    train_ds = dataset_builder_fn(
        config["dataset"]["train_file_name"], epoch=config["train"]["epoch"],
        image_arg_dict=config["dataset"]["image_arg_dict"])

    val_ds = dataset_builder_fn(config["dataset"]["val_file_name"],
                                epoch=None, image_arg_dict=None)
    test_ds = dataset_builder_fn(config["dataset"]["test_file_name"],
                                 epoch=None, image_arg_dict=None)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=config["train"]["learning_rate"])

    anchor_num_per_output = len(
        config["anchor"]["scales"]) * len(config["anchor"]["aspect_ratio"])

    # num_classes + 1 is to include negative class.
    od_model = model_builder.ObjectDetectionModel(
        config["network"]["base_filter_num"],
        config["network"]["kernel_size"], config["network"]["strides"],
        config["network"]["layer_repeat_num"], config["dataset"]["num_classes"],
        anchor_num_per_output)

    compute_loss_fn = functools.partial(
        model_builder.compute_loss, num_classes=config["dataset"]["num_classes"],
        c_weight=config["train"]["classificaiton_loss_weight"],
        r_weight=config["train"]["regression_loss_weight"],
        neg_label_value=config["dataset"]["neg_label_value"],
        ignore_label_value=config["dataset"]["ignore_label_value"],
        negative_ratio=config["train"]["negative_ratio"])

    train_loss_sum = 0
    for train_index, train_item in enumerate(train_ds):
        with tf.GradientTape() as tape:
            train_network_output = od_model(train_item["image"], training=True)
            train_loss = compute_loss_fn(
                train_network_output, train_item["bboxes_preprocessed"], train_item["labels_preprocessed"])
            train_loss_sum += train_loss

            grads = tape.gradient(train_loss, od_model.variables)
            optimizer.apply_gradients(zip(grads, od_model.variables),
                                      global_step=tf.train.get_or_create_global_step())

        if train_index % config["train"]["val_iter"] == 0:
            val_loss_sum = 0
            for val_index, val_item in enumerate(val_ds):
                if val_index != 0 and val_index % config["train"]["val_batch"] == 0:
                    break
                val_network_output = od_model(val_item["image"], training=False)
                val_loss = compute_loss_fn(
                    val_network_output, val_item["bboxes_preprocessed"], val_item["labels_preprocessed"])
                val_loss_sum += val_loss

            train_loss = train_loss_sum / config["train"]["val_iter"]
            val_loss = val_loss_sum / config["train"]["val_batch"]

            add_item_summary(train_item, train_network_output,
                             train_summary_writer, anchors, config["dataset"]["num_classes"])
            add_item_summary(val_item, val_network_output,
                             val_summary_writer, anchors, config["dataset"]["num_classes"])

            add_scalar_summary([train_loss], ["train_loss"],
                               train_summary_writer)
            add_scalar_summary([val_loss], ["val_loss"], val_summary_writer)

            print("Loss at step {:04d}: train loss: {:.3f}, val loss: {:3f}".format(
                train_index, train_loss, val_loss))

            train_loss_sum = 0

        if train_index != 0 and train_index % config["test"]["test_iter"] == 0:
            for test_index, test_item in enumerate(test_ds):
                if (config["test"]["test_batch"] is not None and
                        test_index % config["test"]["test_batch"]):
                    break
                test_network_output = od_model(
                    test_item["image"], training=False)

            bbox_list, label_list = model_builder.predict(
                test_network_output, mask=test_item["mask"], score_threshold=config["test"]["score_threshold"],
                neg_label_value=config["dataset"]["neg_label_value"], anchors=anchors,
                max_prediction=config["test"]["max_prediction"],
                num_classes=config["dataset"]["num_classes"])

            image_list = []
            for image, bbox, label in zip(test_item["image"], bbox_list, label_list):
                normalized_bboxes = normalizing_bboxes(
                    bbox, config["dataset"]["input_shape_h"],
                    config["dataset"]["input_shape_w"])
                image_with_bboxes = tf.image.draw_bounding_boxes(
                    tf.image.convert_image_dtype(
                        image[tf.newaxis, ...], tf.float32),
                    normalized_bboxes[tf.newaxis, ...])
                image_list.append(image_with_bboxes)
            batch_image_tensor = tf.concat(image_list, axis=0)
            with test_summary_writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device("/cpu:0"):
                tf.contrib.summary.image(
                    "image_with_bboxes", batch_image_tensor)
