import tensorflow as tf

tf.enable_eager_execution()

image_name = "../data/000055_128.jpg"
image_string = tf.read_file(image_name)
image = tf.image.decode_jpeg(image_string)[:, :, 0]
print(image[10, 10])
print(image[20, 20])

index = tf.convert_to_tensor([[10, 10], [20, 20]])

selected = tf.gather_nd(image, index)
print(selected)
# print(image[index])
