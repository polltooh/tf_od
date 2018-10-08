def get_output_shape(input_size, kernel_size, strides, layer_repeat_num):
    for num in xrange(layer_repeat_num):
        half = (kernel_size - 1) / 2
        input_size = int((input_size - half) / strides)
    return input_size
