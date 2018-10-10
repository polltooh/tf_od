import functools

import tensorflow as tf
import bbox_lib


def get_output_shape(input_size, kernel_size, strides, layer_repeat_num):
    for num in xrange(layer_repeat_num):
        half = (kernel_size - 1) / 2
        input_size = int((input_size - half) / strides)
    return input_size


def add_item_summary(item, network_output, writer, anchors, num_classes):
    with writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device("/cpu:0"):
        image = item["image"]
        bboxes = item["bboxes"]
        batch_size, image_height, image_width, _ = image.get_shape().as_list()

        normalized_bboxes = bbox_lib.normalizing_bbox(
            bboxes, image_height, image_width)
        image_with_bboxes = tf.image.draw_bounding_boxes(
            tf.image.convert_image_dtype(image, tf.float32), normalized_bboxes)
        tf.contrib.summary.image('image_with_bboxes', image_with_bboxes)

        bboxes_preprocessed = item["bboxes_preprocessed"]
        convert_fn = functools.partial(
            bbox_lib.decode_box_with_anchor, anchors=anchors)
        bboxes_decoded = tf.map_fn(convert_fn, bboxes_preprocessed)
        bboxes_decoded_norm = bbox_lib.normalizing_bbox(
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
