import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

annotation = [{"image_name": "image1",
               "bboxes": [[1, 1, 1, 1],
                          [2, 2, 2, 2]],
               "class": ["ped", "car"]},
              {"image_name": "image2",
               "bboxes": [[3, 3, 3, 3]],
               "class": ["ped"]},
              {"image_name": "image3",
               "bboxes": [[4, 4, 4, 4]],
               "class": ["car"]}]

label_map = {"ped": 0, "car": 1}
filename = "../data/tfrecord.tf"


def check_list(item):
    if not isinstance(item, list):
        item = [item]
    return item


def encode_float(item):
    item = check_list(item)
    return tf.train.Feature(float_list=tf.train.FloatList(value=item))


def encode_bytes(item):
    item = check_list(item)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=item))


def encode_int64(item):
    item = check_list(item)
    check_list(item)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=item))


with tf.python_io.TFRecordWriter(filename) as writer:
    for annot in annotation:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_name": encode_bytes(annot["image_name"]),
                    "bboxes": encode_float(np.array(annot["bboxes"]).reshape(-1).tolist()),
                    "classes": encode_int64([label_map[class_name] for class_name in annot["class"]])
                }))
        writer.write(example.SerializeToString())
