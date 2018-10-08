import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


def convert_to_dense(tensor):
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse_tensor_to_dense(tensor)
    return tensor

def parser(record):
    keys_to_features = {
        'image_name': tf.FixedLenFeature([], dtype=tf.string),
        'bboxes': tf.VarLenFeature(dtype=tf.float32),
        'classes': tf.VarLenFeature(dtype=tf.int64)
    }

    parsed = tf.parse_single_example(record, features=keys_to_features)
    parsed = {key: convert_to_dense(value) for key, value in parsed.iteritems()}

    image_name = parsed["image_name"]
    image_string = tf.read_file(image_name)
    image_decoded = tf.image.decode_jpeg(image_string)

    bboxes = tf.cast(parsed['bboxes'], tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 4))
    classes = parsed["classes"]

    return {"image": image_decoded, "bboxes": bboxes, "classes": classes}

def box_encoding(output_h, output_w):
    anchor = anchor_lib.anchor_gen(output_h, output_w, scales=(1.0), aspect_ratios=(1.0))

def read_data(file_name):
    if not isinstance(file_name, list):
        file_name = [file_name]
    ds = tf.data.TFRecordDataset(file_name)
    ds = ds.map(parser)
    return ds

# train_data = read_data("../data/tfrecord.tf")
# for item in train_data:
#     print(item)
#     # print(item["image_name"].numpy())
#     # print(item["bboxes"].numpy())
