import tensorflow as tf
import numpy as np
import json

tf.enable_eager_execution()


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


if __name__ == "__main__":
    annotation = json.load(open('../file_list/164.json', 'r'))
    train_ratio = 0.8
    train_len = int(len(annotation) * train_ratio)
    train_filename = "../data/164_train.tf"
    val_filename = "../data/164_val.tf"
    write_partition_tf(annotation[:train_len], train_filename)
    write_partition_tf(annotation[train_len:], val_filename)
