import tensorflow as tf

import matcher
import anchor_lib

tf.enable_eager_execution()

class MatcherUtilTest(tf.test.TestCase):
    def test_matching_bbox_label(self):
        bboxes = tf.convert_to_tensor([[10, 10, 30, 30], [30, 30, 50, 50]], tf.float32)
        labels = tf.convert_to_tensor([1, 2], tf.int64)

        anchors = anchor_lib.anchor_gen(4, 4, scales=(1.0), aspect_ratios=(1.0),
                base_anchor_size=(20, 20), anchor_stride=[20, 20])

        bboxes, labels = matcher.matching_bbox_label(bboxes, labels, anchors, 0.7, 0.3, -1, -2)
        expected_labels = tf.convert_to_tensor(
                [-1, -1, -1, -1, -1,  1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1], dtype=tf.int64)
        self.assertAllEqual(labels, expected_labels)

        bboxes = tf.convert_to_tensor([[10, 10, 30, 30], [30, 30, 50, 50]], tf.float32)
        labels = tf.convert_to_tensor([1, 2], tf.int64)
        anchors = anchor_lib.anchor_gen(14, 21, scales=(1.0), aspect_ratios=(1.0),
                base_anchor_size=(20, 20), anchor_stride=[20, 20])
        bboxes, labels = matcher.matching_bbox_label(bboxes, labels, anchors, 0.7, 0.3, -1, -2)
        print(labels)


if __name__ == '__main__':
  tf.test.main()
