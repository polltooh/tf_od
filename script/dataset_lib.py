def convert_to_dense(self, tensor):
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
    parsed = {key: self.convert_to_dense(value) for key, value in parsed.iteritems()}

    image_name = parsed["image_name"]
    image_string = tf.read_file(image_name)
    image_decoded = tf.image.decode_jpeg(image_string)

    bboxes = tf.cast(parsed['bboxes'], tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 4))
    classes = parsed["classes"]

    return {"image": image_decoded, "bboxes": bboxes, "classes": classes}

def preprocess_data(record, record):
    bboxes = record["bboxes"]
    bboxes_encoded = bbox_lib.encode_box_with_anchor(bboxes, anchor)
    record["bboxes"] = bboxes_encoded
    return record

