import functools

import tensorflow as tf

import anchor_lib
import bbox_lib
import matcher


def convert_to_dense(tensor):
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse_tensor_to_dense(tensor)
    return tensor


def parser(record):
    keys_to_features = {
        'image_name': tf.FixedLenFeature([], dtype=tf.string),
        'mask_name': tf.FixedLenFeature([], dtype=tf.string),
        'bboxes': tf.VarLenFeature(dtype=tf.float32),
        'labels': tf.VarLenFeature(dtype=tf.int64)
    }

    parsed = tf.parse_single_example(record, features=keys_to_features)
    parsed = {key: convert_to_dense(value) for key, value in parsed.iteritems()}

    image_name = parsed["image_name"]
    image_string = tf.read_file(image_name)
    image_decoded = tf.image.decode_jpeg(image_string)

    bboxes = tf.cast(parsed['bboxes'], tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 4))

    labels = parsed["labels"]

    return {"image": image_decoded, "bboxes": bboxes, "labels": labels}


def preprocess_data(record, anchors, pos_iou_threshold, neg_iou_threshold,
                    neg_label_value, ignore_label_value):
    bboxes = record["bboxes"]
    labels = record["labels"]

    bboxes, labels = matcher.matching_bbox_label(
        bboxes, labels, anchors, pos_iou_threshold=pos_iou_threshold,
        neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value, ignore_label_value=ignore_label_value)
    bboxes_encoded = bbox_lib.encode_box_with_anchor(bboxes, anchors)
    record["bboxes_preprocessed"] = bboxes_encoded
    record["labels_preprocessed"] = labels
    return record


def read_data(file_name, anchors, epoch, batch_size, pos_iou_threshold, neg_iou_threshold,
              neg_label_value, ignore_label_value):

    ds = tf.data.TFRecordDataset(file_name)
    ds = ds.map(parser)
    preprocess_data_anchor = functools.partial(
        preprocess_data, anchors=anchors,
        pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value, ignore_label_value=ignore_label_value)

    ds = ds.map(preprocess_data_anchor)
    ds = ds.repeat(epoch)
    padded_shapes = {"bboxes": [None, 4], "labels": [None], "image": [None, None, 3],
                     "bboxes_preprocessed": [None, None], "labels_preprocessed": [None]}
    padding_values = {"bboxes": tf.constant(0, tf.float32), "labels": tf.constant(ignore_label_value, tf.int64),
                      "image": tf.constant(0, tf.uint8), "bboxes_preprocessed": tf.constant(ignore_label_value, tf.float32),
                      "labels_preprocessed": tf.constant(0, tf.int64)}

    ds = ds.padded_batch(
        batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
    ds = ds.make_one_shot_iterator()

    return ds
