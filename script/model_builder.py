import tensorflow as tf
import bbox_lib


def hard_negative_loss_mining(c_loss, negative_mask, k):
    """Hard negative mining in classification loss."""
    # make sure at least one negative example
    k = tf.maximum(k, 1)
    # make sure at most all negative.
    k = tf.minimum(k, c_loss.shape[-1])
    neg_c_loss = c_loss * negative_mask
    neg_c_loss = tf.nn.top_k(neg_c_loss, k)[0]
    return tf.reduce_sum(neg_c_loss)


def compute_loss(network_output, bboxes, labels, num_classes, c_weight, r_weight,
                 neg_label_value, ignore_label_value, negative_ratio):
    """Compute loss function."""

    with tf.variable_scope("losses"):
        batch_size = bboxes.shape[0].value
        one_hot_labels = tf.one_hot(labels + 1, num_classes + 1)
        negative_mask = tf.cast(tf.equal(labels, neg_label_value), tf.float32)
        positive_mask = tf.cast(tf.logical_and(tf.not_equal(labels, ignore_label_value),
                                               tf.not_equal(labels, neg_label_value)), tf.float32)

        with tf.variable_scope("classification_loss"):
            classification_output = network_output["classification_output"]
            classification_output = tf.reshape(
                classification_output, [batch_size, -1, num_classes + 1])

            c_loss = tf.losses.softmax_cross_entropy(
                one_hot_labels, classification_output, reduction=tf.losses.Reduction.NONE)

            num_positive = tf.cast(tf.reduce_sum(positive_mask), tf.int32)
            pos_c_loss = tf.reduce_sum(c_loss * positive_mask)
            neg_c_loss = hard_negative_loss_mining(c_loss, negative_mask,
                                                   num_positive * negative_ratio)

            c_loss = (pos_c_loss + neg_c_loss) / batch_size

        with tf.variable_scope("regression_loss"):
            regression_output = network_output["regression_output"]
            regression_output = tf.reshape(
                regression_output, [batch_size, -1, 4])
            r_loss = tf.losses.huber_loss(regression_output, bboxes, delta=1,
                                          reduction=tf.losses.Reduction.NONE)

            r_loss = tf.reduce_sum(
                r_loss * positive_mask[..., tf.newaxis]) / batch_size

        return c_weight * c_loss + r_weight * r_loss


def predict(network_output, mask, score_threshold, neg_label_value, anchors,
            max_prediction, num_classes):
    """Decode predictions from the neural network."""

    classification_output = network_output['classification_output']
    batch_size, _, _, output_dim = classification_output.get_shape().as_list()
    regression_output = network_output['regression_output']
    bbox_list = []
    label_list = []

    ay, ax, ah, aw = bbox_lib.get_center_coordinates_and_sizes(anchors)
    anchor_center_index = tf.cast(tf.transpose(tf.stack([ay, ax])), tf.int32)
    for single_classification_output, single_regression_output, single_mask in zip(
            classification_output, regression_output, mask):
        # num_classes + 1 due to the negative class.
        single_classification_output = tf.reshape(
            single_classification_output, [-1, num_classes + 1])
        single_classification_output = tf.nn.softmax(
            single_classification_output, -1)

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
        predicted_bbox = bbox_lib.decode_box_with_anchor(
            single_regression_output, anchors)
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


def get_output_shape(input_size, kernel_size, strides, layer_repeat_num):
    """Calculate the output shape of the network."""
    for num in xrange(layer_repeat_num):
        half = (kernel_size - 1) / 2
        input_size = int((input_size - half) / strides)
    return input_size


def build_model(num_classes, anchor_num_per_output):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    classification_branch = tf.keras.layers.Conv2D(
            (num_classes + 1) * anchor_num_per_output, (1, 1))(vgg.output)
    regression_branch = tf.keras.layers.Conv2D(
            4 * anchor_num_per_output, (1, 1))(vgg.output)
    model_outputs = [classification_branch, regression_branch]
    return tf.keras.models.Model(vgg.input, model_outputs)


class ObjectDetectionModel(tf.keras.Model):
    def __init__(self, base_filter_num, kernel_size, strides, repeat_num, num_classes,
                 anchor_num_per_output):
        super(ObjectDetectionModel, self).__init__()
        with tf.variable_scope("object_detection_model"):
            params_1 = {'kernel_size': [kernel_size, kernel_size],
                        'strides': strides,
                        'padding': 'valid'}

            params_2 = {'kernel_size': [kernel_size, kernel_size],
                        'strides': 1,
                        'padding': 'same'}
            conv_layers_1 = []
            conv_layers_2 = []
            bn_layers = []
            for layer_num in range(repeat_num):
                conv_layers_1.append(tf.keras.layers.Conv2D(
                    base_filter_num * (layer_num + 1), **params_1))
                conv_layers_2.append(tf.keras.layers.Conv2D(
                    base_filter_num * (layer_num + 1), **params_2))
                bn_layers.append(tf.keras.layers.BatchNormalization())

            self.conv_layers_1 = conv_layers_1
            self.conv_layers_2 = conv_layers_2
            self.bn_layers = bn_layers
            leaky_relu_alpha = 0.3
            self.activation = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)
            dropout_rate = 0.5
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

            self.classification_branch = tf.keras.layers.Conv2D(
                (num_classes + 1) * anchor_num_per_output, (1, 1))

            self.regression_branch = tf.keras.layers.Conv2D(
                4 * anchor_num_per_output, (1, 1))

    def call(self, input_image_tensor, training):
        """Run the model."""
        if input_image_tensor.dtype != tf.float32:
            input_image_tensor = tf.image.convert_image_dtype(
                input_image_tensor, tf.float32)

        h = input_image_tensor
        for conv1, conv2, bn in zip(self.conv_layers_1, self.conv_layers_2, self.bn_layers):
            h = conv1(h)
            h = conv2(h)
            h = bn(h, training=training)
            h = self.activation(h)

        h = self.dropout(h, training=training)

        classification_output = self.classification_branch(h)
        regression_output = self.regression_branch(h)
        return {'classification_output': classification_output,
                'regression_output': regression_output}
