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

    output_h = 15
    output_w = 22
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
        shuffle_buffer_size=config["train"]["shuffle_buffer_size"],
        image_arg_dict=config["dataset"]["image_arg_dict"])

    val_ds = dataset_builder_fn(config["dataset"]["val_file_name"])
    test_ds = dataset_builder_fn(config["dataset"]["test_file_name"])

    decayed_lr = tf.train.cosine_decay(
        learning_rate=config["train"]["learning_rate"],
        global_step=global_step, decay_steps=config['train']['decay_step'],
        alpha=config['train']['decay_alpha'])

    optimizer = tf.train.AdamOptimizer(decayed_lr)

    anchor_num_per_output = len(
        config["anchor"]["scales"]) * len(config["anchor"]["aspect_ratio"])

    od_model = model_builder.build_model(
        config["dataset"]["num_classes"],
        anchor_num_per_output)

    if config["save"]["load_model"]:
        od_model.load_weights(config["save"]["model_dir"])

    compute_loss_fn = functools.partial(
        model_builder.compute_loss, num_classes=config["dataset"]["num_classes"],
        c_weight=config["train"]["classificaiton_loss_weight"],
        r_weight=config["train"]["regression_loss_weight"],
        neg_label_value=config["dataset"]["neg_label_value"],
        ignore_label_value=config["dataset"]["ignore_label_value"],
        negative_ratio=config["train"]["negative_ratio"])

    train_loss_sum = 0
    train_c_loss_sum = 0
    train_r_loss_sum = 0
    for train_index, train_item in enumerate(train_ds):
        with tf.GradientTape() as tape:
            train_network_output = od_model(train_item["image"], training=True)
            train_loss, train_c_loss, train_r_loss = compute_loss_fn(
                train_network_output, train_item["bboxes_preprocessed"],
                train_item["labels_preprocessed"])
            train_loss_sum += train_loss
            train_c_loss_sum += train_c_loss
            train_r_loss_sum += train_r_loss

            grads = tape.gradient(train_loss, od_model.variables)
            optimizer.apply_gradients(zip(grads, od_model.variables),
                                      global_step=tf.train.get_or_create_global_step())

        if train_index != 0 and train_index % config["train"]["val_iter"] == 0:
            val_loss_sum = 0
            val_c_loss_sum = 0
            val_r_loss_sum = 0
            for val_index, val_item in enumerate(val_ds):
                if val_index != 0 and val_index % config["train"]["val_batch"] == 0:
                    break
                val_network_output = od_model(val_item["image"], training=False)
                val_loss, val_c_loss, val_r_loss = compute_loss_fn(
                    val_network_output, val_item["bboxes_preprocessed"],
                    val_item["labels_preprocessed"])
                val_loss_sum += val_loss
                val_c_loss_sum += val_c_loss
                val_r_loss_sum += val_r_loss

            train_loss = train_loss_sum / config["train"]["val_iter"]
            train_c_loss = train_c_loss_sum / config["train"]["val_iter"]
            train_r_loss = train_r_loss_sum / config["train"]["val_iter"]

            val_loss = val_loss_sum / config["train"]["val_batch"]
            val_c_loss = val_c_loss_sum / config["train"]["val_batch"]
            val_r_loss = val_r_loss_sum / config["train"]["val_batch"]

            utils.add_item_summary(train_item, train_network_output,
                                   train_summary_writer, anchors, config["dataset"]["num_classes"])
            utils.add_item_summary(val_item, val_network_output,
                                   val_summary_writer, anchors, config["dataset"]["num_classes"])

            utils.add_scalar_summary(
                [train_loss, train_c_loss, train_r_loss],
                ["loss", "classificaition_loss", "regression_loss"],
                train_summary_writer)

            utils.add_scalar_summary(
                [val_loss, val_c_loss, val_r_loss],
                ["loss", "classificaition_loss", "regression_loss"],
                val_summary_writer)

            print("Loss at step {:04d}: train loss: {:.3f}, val loss: {:3f}".format(
                train_index, train_loss, val_loss))

            train_loss_sum = 0
            train_c_loss_sum = 0
            train_r_loss_sum = 0

        if train_index != 0 and train_index % config["test"]["test_iter"] == 0:
            for test_index, test_item in enumerate(test_ds):
                if test_index != 0 and test_index % config["test"]["test_batch"]:
                    break

                test_network_output = od_model(
                    test_item["image"], training=False)

                bbox_list, label_list = model_builder.predict(
                    test_network_output, mask=test_item["mask"],
                    score_threshold=config["test"]["score_threshold"],
                    neg_label_value=config["dataset"]["neg_label_value"], anchors=anchors,
                    max_prediction=config["test"]["max_prediction"],
                    num_classes=config["dataset"]["num_classes"])

                image_list = []
                for image, bbox, label in zip(test_item["image"], bbox_list, label_list):
                    image += 0.5
                    normalized_bboxes = bbox_lib.normalizing_bbox(
                        bbox, config["dataset"]["input_shape_h"],
                        config["dataset"]["input_shape_w"])
                    image_with_bboxes = tf.image.draw_bounding_boxes(
                        image[tf.newaxis, ...],
                        normalized_bboxes[tf.newaxis, ...])
                    image_list.append(image_with_bboxes)
                batch_image_tensor = tf.concat(image_list, axis=0)
                with test_summary_writer.as_default(), tf.contrib.summary.always_record_summaries(), tf.device("/cpu:0"):
                    tf.contrib.summary.image(
                        "image_with_bboxes", batch_image_tensor)

        if train_index != 0 and train_index % config["save"]["save_per_train_iter"]:
            od_model.save_weights(config["save"]["model_dir"])
