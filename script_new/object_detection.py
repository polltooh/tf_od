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

with open("config.yaml", 'r') as config_file:
    config = yaml.load(config_file)

global_step = tf.train.get_or_create_global_step()

get_output_shape_fn = functools.partial(
    model_builder.get_output_shape, kernel_size=config["network"]["kernel_size"],
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
