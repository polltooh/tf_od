import tensorflow as tf
import bbox_lib


def matching_bbox(bbox1, bbox2, pos_iou_threshold, neg_iou_threshold):
    """ Matching boudning boxes

    Args:
        bbox1: (N, 4) float.
        bbox2: (M, 4) float.
        pos_iou_threshold: iou threshold in order to determine positive match.
        neg_iou_threshold: iou threshold in order to determine negative match.

    Return:
        bbox_iou: (N, M) float.
        matching_index: (M) int. index for matching result.
        pos_match: bool
        neg_match: bool
        ignore_match: bool
    """
    bbox_iou = bbox_lib.iou(bbox1, bbox2)
    match_index = tf.argmax(bbox_iou, 0, output_type=tf.int32)

    matched_vals = tf.reduce_max(bbox_iou, 0)
    neg_match = tf.less(matched_vals, neg_iou_threshold)
    pos_match = tf.greater_equal(matched_vals, pos_iou_threshold)
    ignore_match = tf.logical_and(tf.logical_not(neg_match),
                                  tf.logical_not(pos_match))

    return match_index, pos_match, neg_match, ignore_match


def matching_bbox_label(bboxes, labels, anchors, pos_iou_threshold, neg_iou_threshold,
                        neg_label_value=-1, ignore_label_value=-2):
    """ Matching bounding box and it's labels. It will convert the bboxes to each anchor position.


    """
    match_index, pos_match, neg_match, ignore_match = matching_bbox(
        bboxes, anchors, pos_iou_threshold, neg_iou_threshold)

    labels = tf.gather(labels, match_index)
    bboxes = tf.gather(bboxes, match_index)

    ignore_labels = tf.ones(
        anchors.shape[0].value, tf.int64) * ignore_label_value
    neg_labels = tf.ones(anchors.shape[0].value, tf.int64) * neg_label_value

    labels = tf.where(neg_match, neg_labels, labels)
    labels = tf.where(ignore_match, ignore_labels, labels)

    return bboxes, labels
