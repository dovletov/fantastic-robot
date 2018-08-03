import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import initializers as tfinit

def cnn2d_example(inputs, pkeep_conv, pkeep_hidden):
    """
    """
    print(inputs.shape)
    net = slim.conv2d(inputs = inputs,
                      num_outputs = 64,
                      kernel_size = 3,
                      padding = 'VALID',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tfinit.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tfinit.zeros(),
                      scope = 'conv1'
                      )
    net = slim.dropout(net, pkeep_conv)
    print(net.shape)
    net = slim.conv2d(inputs = net,
                      num_outputs = 128,
                      kernel_size = 3,
                      padding = 'VALID',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tfinit.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tfinit.zeros(),
                      scope='conv2'
                      )
    net = slim.flatten(net)
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 200,
                               scope = 'fc3',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 84,
                               scope = 'fc4',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 16,
                               activation_fn = tf.identity,
                               scope = 'output',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    return net
def cnn3d_example(inputs, pkeep_conv, pkeep_hidden):
    """
    """
    print(inputs.shape)
    net = slim.conv3d(inputs = inputs,
                      num_outputs = 2,
                      kernel_size = 3,
                      padding = 'VALID',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tfinit.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tfinit.zeros(),
                      scope = 'conv1'
                      )
    net = slim.dropout(net, pkeep_conv)
    print(net.shape)
    net = slim.conv3d(inputs = net,
                      num_outputs = 8,
                      kernel_size = 3,
                      padding = 'VALID',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tfinit.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tfinit.zeros(),
                      scope='conv2'
                      )
    # net = tf.squeeze(net, squeeze_dims=[2,3])
    net = slim.flatten(net)
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 200,
                               scope = 'fc3',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 84,
                               scope = 'fc4',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    net = slim.dropout(net, pkeep_hidden)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 16,
                               activation_fn = tf.identity,
                               scope = 'output',
                               weights_initializer = tfinit.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tfinit.zeros()
                               )
    return net
