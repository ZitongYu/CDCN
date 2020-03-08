import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim
import math

def GroupNorm(inputs, is_training=False, activation_fn=None, scope=None, G=16, esp=1e-5):
    '''We force activation_fn as our own hope: tf.nn.relu now!'''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):  
        x = tf.transpose(inputs, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    
    net = output    

    if activation_fn is not None:
        net = activation_fn(net)
    
    #net = tf.nn.relu(output)
    return net

def spatial_gradient_x(intput, name):
    sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_plane_x = np.repeat(sobel_plane_x, intput.get_shape().as_list()[-1], axis=-1)
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_kernel_x = tf.constant(sobel_plane_x, dtype=tf.float32)

    Spatial_Gradient_x = tf.nn.depthwise_conv2d(intput, filter=sobel_kernel_x, \
                                                strides=[1,1,1,1], padding='SAME', name=name+'/spatial_gradient_x')
    return Spatial_Gradient_x

def spatial_gradient_y(intput, name):
    sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_plane_y = np.repeat(sobel_plane_y, intput.get_shape().as_list()[-1], axis=-1)
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_kernel_y = tf.constant(sobel_plane_y, dtype=tf.float32)

    Spatial_Gradient_y = tf.nn.depthwise_conv2d(intput, filter=sobel_kernel_y, \
                                                strides=[1,1,1,1], padding='SAME', name=name+'/spatial_gradient_y')
    return Spatial_Gradient_y

def residual_gradient_conv(input, out_dim, is_training, name, gradient_type='gabor',\
                             is_bn = True, activation_fn = tf.nn.relu):
    '''
        gradient_type: 'gabor', 'sobel', 'normal'
    '''
    #pw = slim.conv2d(input, out_dim, [1,1],stride=[1,1],activation_fn=None,scope=name+'/rgc_pw_1',padding='SAME')
    net = slim.conv2d(input, out_dim, [3,3],stride=[1,1],activation_fn=None,scope=name+'/conv',padding='SAME')
    gradient_x = spatial_gradient_x(input, name)
    gradient_y = spatial_gradient_y(input, name)

    if gradient_type == 'sobel':
        gradient_x_pw = slim.conv2d(gradient_x, out_dim, [1,1],stride=[1,1],activation_fn=None,scope=name+'/rgc_pw_x',padding='SAME')
        gradient_y_pw = slim.conv2d(gradient_y, out_dim, [1,1],stride=[1,1],activation_fn=None,scope=name+'/rgc_pw_y',padding='SAME')
        net = net + gradient_x_pw + gradient_y_pw
    elif gradient_type == 'gabor':
        #gradient_gabor = tf.sqrt(tf.pow(gradient_x, 2) + tf.pow(gradient_y, 2))
        gradient_gabor = tf.pow(gradient_x, 2) + tf.pow(gradient_y, 2)
        gradient_gabor_pw = slim.conv2d(gradient_gabor, out_dim, [1,1],stride=[1,1],activation_fn=None,scope=name+'/rgc_pw_gabor',padding='SAME')
        gradient_gabor_pw = slim.batch_norm(gradient_gabor_pw, is_training=is_training, activation_fn=None, scope= name + '/gabor_bn')
        net = net + gradient_gabor_pw
    elif gradient_type == 'normal':
        net = net
    else:
        print('Unknown gradient_type for "residual_gradient_conv"')
        exit(1)
    
    if is_bn:
        #net=GroupNorm(net, is_training=is_training, activation_fn=None, scope= name + '/gn')
        net=slim.batch_norm(net, is_training=is_training, activation_fn=None, scope= name + '/gn')

    if activation_fn is not None:
        net = activation_fn(net)

    return net


def Conv2d_cd(input, filters, kernel_size=3, strides=1,
              padding='SAME', theta=0.7, use_bias=False,
              kernel_initializer=tf.variance_scaling_initializer(scale=2.0),

              name='conv2d_cd'):
    out_channels = filters
    name_scope = name
    if padding=='same':
        padding='SAME'
    if padding=='valid':
        padding='VALID'
    print(input)
    input_shape = input.get_shape().as_list()
    _filter = tf.get_variable(name=name_scope, shape=[kernel_size, kernel_size, input_shape[-1], out_channels], initializer=kernel_initializer)
    out_normal = tf.nn.conv2d(input, _filter, strides=[1, strides, strides, 1], padding=padding, name=name_scope + '/normal')
    
    if math.fabs(theta - 0.0) < 1e-8:
        return out_normal 
    kernel_diff = tf.reduce_sum(_filter, axis=0, keep_dims=True)
    kernel_diff = tf.reduce_sum(kernel_diff, axis=1, keep_dims=True)
    print(name_scope + '/kernel_diff.shape:', kernel_diff.get_shape())
    kernel_diff = tf.tile(kernel_diff, [kernel_size, kernel_size, 1, 1])
    print(name_scope + '/kernel_diff.shape:', kernel_diff.get_shape())
    out_diff = tf.nn.conv2d(input, kernel_diff, strides=[1, strides, strides, 1], padding=padding, name=name_scope + '/diff')

    return out_normal - theta * out_diff

def CDCN_BLOCK(input, out_channels, kernel_size=3, is_training=False, name='conv2d_cd'):
    cdcn_out = Conv2d_cd(input, out_channels, kernel_size=kernel_size, name=name)
    net = slim.batch_norm(cdcn_out, is_training=is_training, activation_fn=None, scope= name + '/bn')
    net = tf.nn.relu(net)
    return net

def contrast_depth_conv(input, dilation_rate = 1, op_name = 'contrast_depth'):
    ''' compute contrast depth in both of (out, label) '''
    assert(input.get_shape()[-1] == 1)

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = np.expand_dims(kernel_filter, axis = -1)
    kernel_filter_tf = tf.constant(kernel_filter, dtype=tf.float32)

    if dilation_rate == 1:
        contrast_depth = tf.nn.conv2d(input, kernel_filter_tf, \
                            strides = [1,1,1,1], padding = 'SAME', name = op_name)
    else:
        contrast_depth = tf.nn.atrous_conv2d(input, kernel_filter_tf, \
                            rate = dilation_rate, padding = 'SAME', name = op_name)
    
    return contrast_depth

def contrast_depth_loss(out, label):
    '''
    compute contrast depth in both of (out, label),
    then get the loss of them
    tf.atrous_convd match tf-versions: 1.4
    '''
    contrast_out = contrast_depth_conv(out, 1, 'contrast_out')
    contrast_label = contrast_depth_conv(label, 1, 'contrast_label')

    loss = tf.pow(contrast_out - contrast_label, 2)
    loss = tf.reduce_mean(loss)

    return loss

    

    
