from __future__ import division
from __future__ import print_function

import functools
import os
import json
import random
import urllib
# import requests
import zipfile

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
EPSILON = 1e-6

"""Download data. """
# def download_and_extract_data(data_url, tmp_dir, cam_num):
#     full_data_url = os.path.join(data_url, cam_num + ".zip")
#     output_path = os.path.join(tmp_dir, cam_num + ".zip")
#     respond = requests.get(full_data_url, allow_redirects=True)
#     with open(output_path, 'wb') as f:
#         f.write(respond.content)
#     zip_ref = zipfile.ZipFile(output_path, 'r')
#     zip_ref.extractall(tmp_dir)
#     zip_ref.close()
# 
# 
# tmp_dir = "/tmp/"
# cam_num = '164'
# data_url = "https://github.com/polltooh/traffic_video_analysis/raw/master/data/"
# download_and_extract_data(data_url, tmp_dir, cam_num)


"""Prepare data. Split and convert to tf records."""
def check_list(item):
    if not isinstance(item, list):
        item = [item]
    return item


def encode_float(item):
    item = check_list(item)
    return tf.train.Feature(float_list=tf.train.FloatList(value=item))


def encode_bytes(item):
    item = check_list(item)
    item = [it.encode('utf-8') for it in item]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=item))


def encode_int64(item):
    item = check_list(item)
    check_list(item)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=item))


def write_partition_tf(annotation, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for annot in annotation:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_name": encode_bytes(annot["image_name"]),
                        "mask_name": encode_bytes(annot["mask_name"]),
                        "bboxes": encode_float(np.array(annot["bboxes"]).reshape(-1).tolist()),
                        "labels": encode_int64(annot["labels"])
                    }))
            writer.write(example.SerializeToString())


def partition_data(tmp_dir, cam_num, train_ratio):
    annotation = json.load(
        open(os.path.join(tmp_dir, cam_num, 'annotation.json'), 'r'))
    random.shuffle(annotation)
    train_len = int(len(annotation) * train_ratio)
    train_filename = os.path.join(tmp_dir, cam_num, "train.tf")
    val_filename = os.path.join(tmp_dir, cam_num, "val.tf")
    write_partition_tf(annotation[:train_len], train_filename)
    write_partition_tf(annotation[train_len:], val_filename)


# train_ratio = 0.8
# partition_data(tmp_dir, cam_num, train_ratio)


"""Preprocessing related functions."""
def expanded_shape(orig_shape, start_dim, num_dims):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def meshgrid(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))
    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid


def tile_anchors(grid_height, grid_width, scales, aspect_ratios,
                 base_anchor_size, anchor_stride, anchor_offset):
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # Get a grid of box centers
    y_centers = tf.to_float(tf.range(grid_height))
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.to_float(tf.range(grid_width))
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bbox_corners


def center_size_bbox_to_corners_bbox(centers, sizes):
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


def anchor_gen(grid_height,
               grid_width,
               scales,
               aspect_ratios,
               base_anchor_size,
               anchor_stride,
               anchor_offset=[0.0, 0.0]):
    base_anchor_size = tf.to_float(tf.convert_to_tensor(base_anchor_size))
    anchor_stride = tf.to_float(tf.convert_to_tensor(anchor_stride))
    anchor_offset = tf.to_float(tf.convert_to_tensor(anchor_offset))

    scales_grid, aspect_ratios_grid = meshgrid(scales, aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(grid_height, grid_width, scales_grid,
                           aspect_ratios_grid, base_anchor_size, anchor_stride,
                           anchor_offset)

    return anchors


# init for the model.
output_h = 15
output_w = 22
input_shape_h = 240
input_shape_w = 352

anchor_strides = [input_shape_h / output_h, input_shape_w / output_w]

anchor_scales = [0.5, 1.0, 2.0]
anchor_aspect_ratio = [0.5, 1.0, 2.0]
anchor_base_anchor_size = [32, 32]

anchors = anchor_gen(
    output_h,
    output_w,
    scales=anchor_scales,
    aspect_ratios=anchor_aspect_ratio,
    base_anchor_size=anchor_base_anchor_size,
    anchor_stride=anchor_strides)

anchor_num_per_output = len(anchor_scales) * len(anchor_aspect_ratio)


num_classes = 10


# init for the data input.
batch_size = 32
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3
neg_label_value = -1
ignore_label_value = -2



def intersection(bbox1, bbox2, scope=None):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=bbox1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=bbox2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def area(bbox, scope=None):
    y_min, x_min, y_max, x_max = tf.split(
        value=bbox, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def iou(bbox1, bbox2, scope=None):
    intersections = intersection(bbox1, bbox2)
    areas1 = area(bbox1)
    areas2 = area(bbox2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


def matched_iou(bbox1, bbox2, scope=None):
    intersections = matched_intersection(bbox1, bbox2)
    areas1 = area(bbox1)
    areas2 = area(bbox2)
    unions = areas1 + areas2 - intersections
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


def get_center_coordinates_and_sizes(box_corners, scope=None):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]


def encode_box_with_anchor(bbox, anchor):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchor)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(bbox)
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    return tf.transpose(tf.stack([ty, tx, th, tw]))


def decode_box_with_anchor(encoded_bbox, anchor):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchor)
    ty, tx, th, tw = tf.unstack(tf.transpose(encoded_bbox))
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def normalizing_bbox(bbox, image_height, image_width):
    if tf.rank(bbox).numpy() == 2:
        axis = 1
    elif tf.rank(bbox).numpy() == 3:
        axis = 2

    y_min, x_min, y_max, x_max = tf.split(
        value=bbox, num_or_size_splits=4, axis=axis)

    normalized_bbox = tf.concat([
        y_min / image_height, x_min / image_width, y_max / image_height,
        x_max / image_width
    ],
        axis=axis)
    return normalized_bbox


def convert_to_dense(tensor):
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse_tensor_to_dense(tensor)
    return tensor


def matching_bbox(bbox, anchors, pos_iou_threshold, neg_iou_threshold):
    bbox_iou = iou(bbox, anchors)
    match_index = tf.argmax(bbox_iou, 0, output_type=tf.int32)

    matched_vals = tf.reduce_max(bbox_iou, 0)
    neg_match = tf.less(matched_vals, neg_iou_threshold)
    pos_match = tf.greater_equal(matched_vals, pos_iou_threshold)
    ignore_match = tf.logical_and(
        tf.logical_not(neg_match), tf.logical_not(pos_match))

    # Finding the best matches for unmatched bboxes.
    non_pos_match = tf.logical_not(pos_match)
    bbox_iou = bbox_iou * tf.cast(non_pos_match[tf.newaxis, ...], tf.float32)

    match_index_anchor_to_bbox = tf.argmax(bbox_iou, 1, output_type=tf.int32)
    anchor_num = anchors.get_shape().as_list()[0]  # M
    match_index_anchor_to_bbox_one_hot = tf.one_hot(match_index_anchor_to_bbox,
                                                    anchor_num)
    match_index_anchor_to_bbox_indices = tf.argmax(
        match_index_anchor_to_bbox_one_hot, 0, output_type=tf.int32)  # M
    match_index_anchor_to_bbox_mask = tf.cast(
        tf.reduce_max(match_index_anchor_to_bbox_one_hot, 0), tf.bool)  # M

    match_index = tf.where(match_index_anchor_to_bbox_mask,
                           match_index_anchor_to_bbox_indices, match_index)
    pos_match = tf.logical_or(match_index_anchor_to_bbox_mask, pos_match)
    neg_match = tf.logical_and(tf.logical_not(pos_match), neg_match)
    ignore_match = tf.logical_and(tf.logical_not(pos_match), ignore_match)

    return match_index, pos_match, neg_match, ignore_match


def matching_bbox_label(bboxes,
                        labels,
                        anchors,
                        pos_iou_threshold,
                        neg_iou_threshold,
                        neg_label_value=-1,
                        ignore_label_value=-2):
    match_index, pos_match, neg_match, ignore_match = matching_bbox(
        bboxes, anchors, pos_iou_threshold, neg_iou_threshold)

    labels_encoded = tf.gather(labels, match_index)
    bboxes_encoded = tf.gather(bboxes, match_index)

    ignore_labels = tf.ones(
        anchors.shape[0].value, tf.int64) * ignore_label_value

    neg_labels = tf.ones(anchors.shape[0].value, tf.int64) * neg_label_value

    labels_encoded = tf.where(neg_match, neg_labels, labels_encoded)
    labels_encoded = tf.where(ignore_match, ignore_labels, labels_encoded)

    return bboxes_encoded, labels_encoded


def parser(record):
    keys_to_features = {
        "image_name": tf.FixedLenFeature([], dtype=tf.string),
        "mask_name": tf.FixedLenFeature([], dtype=tf.string),
        "bboxes": tf.VarLenFeature(dtype=tf.float32),
        "labels": tf.VarLenFeature(dtype=tf.int64)
    }

    parsed = tf.parse_single_example(record, features=keys_to_features)
    parsed = {key: convert_to_dense(value) for key, value in parsed.iteritems()}

    # tmp_d = tf.convert_to_tensor(tmp_dir, tf.string)
    # image_name = tf.strings.join([tmp_dir, parsed["image_name"]])
    image_name = parsed["image_name"]
    image_string = tf.read_file(image_name)

    image_decoded = tf.image.decode_jpeg(image_string)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

    mask_name = parsed["mask_name"]
    mask_string = tf.read_file(mask_name)
    mask_decoded = tf.image.decode_png(mask_string)
    # Reducing mask to two dimention.
    mask_decoded = tf.reduce_any(tf.cast(mask_decoded, tf.bool), -1)

    bboxes = tf.cast(parsed["bboxes"], tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 4))

    labels = parsed["labels"]

    return {
        "image": image_decoded,
        "mask": mask_decoded,
        "bboxes": bboxes,
        "labels": labels
    }


def image_argumentation(image):
    image = tf.image.random_brightness(image, 0.125)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image


def preprocess_data(record,
                    anchors,
                    pos_iou_threshold,
                    neg_iou_threshold,
                    neg_label_value,
                    ignore_label_value,
                    image_arg=False):

    if image_arg:
        record["image"] = image_argumentation(record["image"])

    # Whiten the image.
    record["image"] -= 0.5

    bboxes = record["bboxes"]
    labels = record["labels"]

    bboxes, labels = matching_bbox_label(
        bboxes,
        labels,
        anchors,
        pos_iou_threshold=pos_iou_threshold,
        neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value,
        ignore_label_value=ignore_label_value)

    mask = record["mask"]
    # ignore the indices outside of the mask.
    ay, ax, ah, aw = get_center_coordinates_and_sizes(anchors)
    anchor_center_index = tf.cast(tf.transpose(tf.stack([ay, ax])), tf.int32)
    in_mask = tf.gather_nd(mask, anchor_center_index)
    ignore_labels = tf.ones_like(labels) * ignore_label_value
    labels = tf.where(in_mask, labels, ignore_labels)

    bboxes_encoded = encode_box_with_anchor(bboxes, anchors)
    record["bboxes_preprocessed"] = bboxes_encoded
    record["labels_preprocessed"] = labels
    return record


def read_data(file_name,
              anchors,
              batch_size,
              pos_iou_threshold,
              neg_iou_threshold,
              neg_label_value,
              ignore_label_value,
              epoch=None,
              shuffle_buffer_size=None,
              image_arg=False):

    ds = tf.data.TFRecordDataset(file_name)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.repeat(epoch)

    ds = ds.map(parser)
    preprocess_data_anchor = functools.partial(
        preprocess_data,
        anchors=anchors,
        pos_iou_threshold=pos_iou_threshold,
        neg_iou_threshold=neg_iou_threshold,
        neg_label_value=neg_label_value,
        ignore_label_value=ignore_label_value,
        image_arg=image_arg)

    ds = ds.map(preprocess_data_anchor, num_parallel_calls=4)
    padded_shapes = {
        "bboxes": [None, 4],
        "labels": [None],
        "image": [None, None, 3],
        "mask": [None, None],
        "bboxes_preprocessed": [None, None],
        "labels_preprocessed": [None]
    }
    padding_values = {
        "bboxes": tf.constant(0, tf.float32),
        "labels": tf.constant(ignore_label_value, tf.int64),
        "image": tf.constant(0, tf.float32),
        "mask": tf.constant(False, tf.bool),
        "bboxes_preprocessed": tf.constant(ignore_label_value, tf.float32),
        "labels_preprocessed": tf.constant(0, tf.int64)
    }

    ds = ds.padded_batch(
        batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
    ds = ds.prefetch(batch_size * 5)
    ds = ds.make_one_shot_iterator()

    return ds


"""Build the dataset."""
dataset_builder_fn = functools.partial(
    read_data,
    anchors=anchors,
    batch_size=batch_size,
    pos_iou_threshold=pos_iou_threshold,
    neg_iou_threshold=neg_iou_threshold,
    neg_label_value=neg_label_value,
    ignore_label_value=ignore_label_value)

train_file_name = os.path.join("/home/guanhangwu/tf_od/tutorial/data/164/train.tf")
val_file_name = os.path.join("/home/guanhangwu/tf_od/tutorial/data/164/val.tf")
test_file_name = os.path.join("/home/guanhangwu/tf_od/tutorial/data/164/val.tf")

epoch = 30
shuffle_buffer_size = 1000

train_ds = dataset_builder_fn(
    train_file_name,
    epoch=epoch,
    shuffle_buffer_size=shuffle_buffer_size,
    image_arg=True)

val_ds = dataset_builder_fn(val_file_name)
test_ds = dataset_builder_fn(test_file_name)

"""Build the model."""
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


od_model = build_model(num_classes, anchor_num_per_output)


"""Learning related params and functions."""
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



def hard_negative_loss_mining(c_loss, negative_mask, k):
    """Hard negative mining in classification loss."""
    # make sure at least one negative example
    k = tf.maximum(k, 1)
    # make sure at most all negative.
    k = tf.minimum(k, c_loss.shape[-1])
    neg_c_loss = c_loss * negative_mask
    neg_c_loss = tf.nn.top_k(neg_c_loss, k)[0]
    return tf.reduce_sum(neg_c_loss)


def compute_loss(network_output, bboxes, labels, num_classes, c_weight,
                 r_weight, neg_label_value, ignore_label_value, negative_ratio):
    """Compute loss function."""

    with tf.variable_scope("losses"):
        batch_size = bboxes.shape[0].value
        one_hot_labels = tf.one_hot(labels + 1, num_classes + 1)
        negative_mask = tf.cast(tf.equal(labels, neg_label_value), tf.float32)
        positive_mask = tf.cast(
            tf.logical_and(
                tf.not_equal(labels, ignore_label_value),
                tf.not_equal(labels, neg_label_value)), tf.float32)

        with tf.variable_scope("classification_loss"):
            classification_output = network_output[0]
            classification_output = tf.reshape(classification_output,
                                               [batch_size, -1, num_classes + 1])

            c_loss = tf.losses.softmax_cross_entropy(
                one_hot_labels,
                classification_output,
                reduction=tf.losses.Reduction.NONE)

            num_positive = tf.cast(tf.reduce_sum(positive_mask), tf.int32)
            pos_c_loss = tf.reduce_sum(c_loss * positive_mask)
            neg_c_loss = hard_negative_loss_mining(c_loss, negative_mask,
                                                   num_positive * negative_ratio)

            c_loss = (pos_c_loss + neg_c_loss) / batch_size

        with tf.variable_scope("regression_loss"):
            regression_output = network_output[1]
            regression_output = tf.reshape(
                regression_output, [batch_size, -1, 4])
            r_loss = tf.losses.huber_loss(
                regression_output,
                bboxes,
                delta=1,
                reduction=tf.losses.Reduction.NONE)

            r_loss = tf.reduce_sum(
                r_loss * positive_mask[..., tf.newaxis]) / batch_size

        return c_weight * c_loss + r_weight * r_loss, c_loss, r_loss


def predict(network_output, mask, score_threshold, neg_label_value, anchors,
            max_prediction, num_classes):
    """Decode predictions from the neural network."""

    classification_output = network_output[0]
    batch_size, _, _, output_dim = classification_output.get_shape().as_list()
    regression_output = network_output[1]
    bbox_list = []
    label_list = []

    ay, ax, ah, aw = get_center_coordinates_and_sizes(anchors)
    anchor_center_index = tf.cast(tf.transpose(tf.stack([ay, ax])), tf.int32)
    for single_classification_output, single_regression_output, single_mask in zip(
            classification_output, regression_output, mask):
        # num_classes + 1 due to the negative class.
        single_classification_output = tf.reshape(single_classification_output,
                                                  [-1, num_classes + 1])
        single_classification_output = tf.nn.softmax(single_classification_output,
                                                     -1)

        max_confidence = tf.reduce_max(single_classification_output, -1)
        confident_mask = max_confidence > score_threshold
        # - 1 due to the negative class.
        max_index = tf.argmax(single_classification_output, 1) - 1
        non_negative_mask = tf.not_equal(max_index, -1)
        in_mask = tf.gather_nd(single_mask, anchor_center_index)
        foreground_mask = tf.logical_and(
            in_mask, tf.logical_and(confident_mask, non_negative_mask))

        valid_labels = tf.boolean_mask(max_index, foreground_mask)

        single_regression_output = tf.reshape(single_regression_output, [-1, 4])
        predicted_bbox = decode_box_with_anchor(single_regression_output,
                                                anchors)
        valid_boxes = tf.boolean_mask(predicted_bbox, foreground_mask)
        valid_confidence_score = tf.boolean_mask(
            max_confidence, foreground_mask)

        selected_indices = tf.image.non_max_suppression(
            valid_boxes, valid_confidence_score, max_prediction)

        valid_boxes = tf.gather(valid_boxes, selected_indices)
        valid_labels = tf.gather(valid_labels, selected_indices)
        bbox_list.append(valid_boxes)
        label_list.append(valid_labels)

    return bbox_list, label_list

# init for the loss.
classificaiton_loss_weight = 1
regression_loss_weight = 10
negative_ratio = 3

compute_loss_fn = functools.partial(
    compute_loss,
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
        train_loss, _, _ = compute_loss_fn(train_network_output,
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
            val_loss, _, _ = compute_loss_fn(val_network_output,
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

            bbox_list, label_list = predict(
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
                normalized_bboxes = normalizing_bbox(
                    bbox, input_shape_h, input_shape_w)
                image_with_bboxes = tf.image.draw_bounding_boxes(
                    image[tf.newaxis, ...], normalized_bboxes[tf.newaxis, ...])
                image_list.append(image_with_bboxes)
