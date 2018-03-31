# compute input size from a given output size
f = lambda output_size, ksize, stride: (output_size - 1) * stride + ksize

out = f(f(f(1, 4, 1),    # conv2 -> out
               4, 1),    # conv1 -> conv2
               4, 2)     # input -> conv1
print('nlayers1 receptive field size: {}'.format(out))

out = f(f(f(f(1, 4, 1),    # conv3 -> out
                 4, 1),    # conv2 -> conv3
                 4, 2),    # conv1 -> conv2
                 4, 2)     # input -> conv1
print('nlayers2 receptive field size: {}'.format(out))

out = f(f(f(f(f(1, 4, 1),    # conv5 -> out
                   4, 1),    # conv3 -> conv4
                   4, 2),    # conv2 -> conv3
                   4, 2),    # conv1 -> conv2
                   4, 2)     # input -> conv1
print('nlayers3 receptive field size: {}'.format(out))

out = f(f(f(f(f(f(1, 4, 1),    # conv5 -> out
                     4, 1),    # conv4 -> conv5
                     4, 2),    # conv3 -> conv4
                     4, 2),    # conv2 -> conv3
                     4, 2),    # conv1 -> conv2
                     4, 2)     # input -> conv1
print('nalyers4 receptive field size: {}'.format(out))

out = f(f(f(f(f(f(f(1, 4, 1),    # conv6 -> out
                       4, 1),    # conv5 -> conv6
                       4, 2),    # conv4 -> conv5
                       4, 2),    # conv3 -> conv4
                       4, 2),    # conv2 -> conv3
                       4, 2),    # conv1 -> conv2
                       4, 2)     # input -> conv1
print('nalyers5 receptive field size: {}'.format(out))
