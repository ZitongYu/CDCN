import tensorflow as tf
from tensorflow.python.platform import flags
from util_network_CDCN import Conv2d_cd

FLAGS = flags.FLAGS



class ZZNet(object):
    def __init__(self):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.img_size = 256
        # self.train_flag = True
        self.bn = tf.layers.batch_normalization
        if FLAGS.leaky_relu:
            self.active = tf.nn.leaky_relu
        else:
            self.active = tf.nn.relu

        if FLAGS.CDC:
            self.conv = Conv2d_cd
        else:
            self.conv = tf.layers.conv2d


    def forward(self, face, depth, IR, training):
        # forward of the representation module
        init = tf.variance_scaling_initializer(scale=2.0)
        inp = tf.concat([face, depth, IR], axis=-1)

        net = self.conv(inp, self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_root')
        net = self.active(net)
        net = self.bn(net, name='bn_init', training=training)

        net = self.conv(net, 2*self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_1')
        net = self.active(net)
        net = self.bn(net, name='bn1_1', training=training)

        net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_2')
        net = self.active(net)
        net = self.bn(net, name='bn1_2', training=training)

        net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_3')
        net = self.active(net)
        net = self.bn(net, name='bn1_3', training=training)

        pool1 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        if FLAGS.attention:
            size = pool1.get_shape().as_list()
            att = tf.nn.avg_pool(pool1, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1], padding='SAME')
            att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,  name='conv_att1')
            pool1 = tf.multiply(pool1, att)

        net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_1')
        net = self.active(net)
        net = self.bn(net, name='bn2_1', training=training)

        net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_2')
        net = self.active(net)
        net = self.bn(net, name='bn2_2', training=training)

        net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_3')
        net = self.active(net)
        net = self.bn(net, name='bn2_3', training=training)

        pool2 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        if FLAGS.attention:
            size = pool2.get_shape().as_list()
            att = tf.nn.avg_pool(pool2, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1], padding='SAME')
            att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,  name='conv_att2')
            pool2 = tf.multiply(pool2, att)

        net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_1')
        net = self.active(net)
        net = self.bn(net, name='bn3_1', training=training)

        net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_2')
        net = self.active(net)
        net = self.bn(net, name='bn3_2', training=training)

        net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_3')
        net = self.active(net)
        net = self.bn(net, name='bn3_3', training=training)

        pool3 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        if FLAGS.attention:
            size = pool3.get_shape().as_list()
            att = tf.nn.avg_pool(pool3, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1], padding='SAME')
            att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,  name='conv_att3')
            pool3 = tf.multiply(pool3, att)

        feature1 = tf.image.resize_bilinear(pool1, size=(32, 32))
        feature2 = tf.image.resize_bilinear(pool2, size=(32, 32))
        feature3 = tf.image.resize_bilinear(pool3, size=(32, 32))

        pool_concat = tf.concat([feature1, feature2, feature3], axis=-1)

        net = self.conv(pool_concat, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_4_1')
        net = self.active(net)
        net = self.bn(net, name='bn4_1', training=training)

        feature = tf.nn.avg_pool(net, [1,8,8,1], [1,8,8,1], padding="SAME")
        feature = tf.layers.flatten(feature)
        if FLAGS.dropout:
            feature = tf.layers.dropout(feature, rate=FLAGS.dropout, training=training)
        fc1 = tf.layers.dense(feature, 128, name='fc1', kernel_initializer=init)
        fc1 = self.active(fc1)
        fc1 = self.bn(fc1, name='bn_binary', training=training)

        if FLAGS.dropout:
            fc1 = tf.layers.dropout(fc1, rate=FLAGS.dropout, training=training)
        fc2 = tf.layers.dense(fc1, 2, name='fc2', kernel_initializer=init)

        net = self.conv(net, self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_4_2')
        net = self.active(net)
        net = self.bn(net, name='bn4_2', training=training)

        if FLAGS.attention:
            size = net.get_shape().as_list()
            att = tf.nn.avg_pool(net, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1], padding='SAME')
            att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,  name='conv_att4')
            net = tf.multiply(net, att)

        net = self.conv(net, 1, 3, padding='same', kernel_initializer=init,  name='conv_4_3')

        if FLAGS.last_bn:
            net = self.bn(net, name='bn_last', training=training)
        if FLAGS.last_relu:
            net = self.active(net)

        return net, fc2

    def forward_split(self, face, depth, IR, training):
        # forward of the representation module
        init = tf.variance_scaling_initializer(scale=2.0)
        init2 = tf.zeros_initializer

        inputs = [face, depth, IR]
        features = []

        # with tf.variable_scope('conv'):
        for i in range(3):
            with tf.variable_scope(str(i)):
                net = self.conv(inputs[i], self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_root')
                net = self.active(net)
                net = self.bn(net, name='bn_init', training=training)

                net = self.conv(net, 2*self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_1')
                net = self.active(net)
                net = self.bn(net, name='bn1_1', training=training)

                net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_2')
                net = self.active(net)
                net = self.bn(net, name='bn1_2', training=training)

                net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_1_3')
                net = self.active(net)
                net = self.bn(net, name='bn1_3', training=training)

                pool1 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                if FLAGS.attention:
                    size = pool1.get_shape().as_list()
                    att = tf.nn.avg_pool(pool1, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                         padding='SAME')
                    att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init, name='conv_att1')
                    pool1 = tf.multiply(pool1, att)


                net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_1')
                net = self.active(net)
                net = self.bn(net, name='bn2_1', training=training)

                net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_2')
                net = self.active(net)
                net = self.bn(net, name='bn2_2', training=training)

                net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_2_3')
                net = self.active(net)
                net = self.bn(net, name='bn2_3', training=training)

                pool2 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                if FLAGS.attention:
                    size = pool2.get_shape().as_list()
                    att = tf.nn.avg_pool(pool2, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                         padding='SAME')
                    att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init, name='conv_att2')
                    pool2 = tf.multiply(pool2, att)

                net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_1')
                net = self.active(net)
                net = self.bn(net, name='bn3_1', training=training)

                net = self.conv(net, 3 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_2')
                net = self.active(net)
                net = self.bn(net, name='bn3_2', training=training)

                net = self.conv(net, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_3_3')
                net = self.active(net)
                net = self.bn(net, name='bn3_3', training=training)

                pool3 = tf.nn.max_pool(net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                if FLAGS.attention:
                    size = pool3.get_shape().as_list()
                    att = tf.nn.avg_pool(pool3, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                         padding='SAME')
                    att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init, name='conv_att3')
                    pool3 = tf.multiply(pool3, att)

                feature1 = tf.image.resize_bilinear(pool1, size=(32, 32))
                feature2 = tf.image.resize_bilinear(pool2, size=(32, 32))
                feature3 = tf.image.resize_bilinear(pool3, size=(32, 32))

                pool_concat = tf.concat([feature1, feature2, feature3], axis=-1)

                net = self.conv(pool_concat, 2 * self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_4_1')
                net = self.active(net)
                net = self.bn(net, name='bn4_1', training=training)

                features.append(net)


        net = tf.concat(features, axis=-1)

        feature = tf.nn.avg_pool(net, [1,8,8,1], [1,8,8,1], padding="SAME")
        feature = tf.layers.flatten(feature)
        if FLAGS.dropout:
            feature = tf.layers.dropout(feature, rate=FLAGS.dropout, training=training)
        fc1 = tf.layers.dense(feature, 128, name='fc1', kernel_initializer=init,  )
        fc1 = self.active(fc1)
        fc1 = self.bn(fc1, name='bn_binary', training=training)

        if FLAGS.dropout:
            fc1 = tf.layers.dropout(fc1, rate=FLAGS.dropout, training=training)
        fc2 = tf.layers.dense(fc1, 2, name='fc2', kernel_initializer=init,  )

        net = self.conv(net, self.dim_hidden, 3, padding='same', kernel_initializer=init,  name='conv_4_2')
        net = self.active(net)
        net = self.bn(net, name='bn4_2', training=training)

        if FLAGS.attention:
            size = net.get_shape().as_list()
            att = tf.nn.avg_pool(net, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                 padding='SAME')
            att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init, name='conv_att4')
            net = tf.multiply(net, att)

        net = self.conv(net, 1, 3, padding='same', kernel_initializer=init,  name='conv_4_3')

        if FLAGS.last_bn:
            net = self.bn(net, name='bn_last', training=training)
        if FLAGS.last_relu:
            net = self.active(net)

        return net, fc2



class DTN(object):
    def __init__(self):
        # self.train_flag = True
        if FLAGS.leaky_relu:
            self.active = tf.nn.leaky_relu
        else:
            self.active = tf.nn.relu

        if FLAGS.CDC:
            self.conv = Conv2d_cd
        else:
            self.conv = tf.layers.conv2d


    def forward(self, face, depth, IR, training=True):

        init = tf.variance_scaling_initializer(scale=2.0)

        root = tf.concat([face, depth, IR], axis=-1)

        root = self.conv(root, filters=FLAGS.base_num_filters, kernel_size=5, padding='same', use_bias=False, name='root_conv')
        root = tf.layers.batch_normalization(root, training=training, name='root_bn')
        root = self.active(root)

        with tf.variable_scope('CRU/1', reuse=False):
            root = self.CRU(root, training=training)

            if FLAGS.attention:
                size = root.get_shape().as_list()
                att = tf.nn.avg_pool(root, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                     padding='SAME')
                att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,
                                        name='conv_att')
                root = tf.multiply(root, att)

        with tf.variable_scope('CRU/2', reuse=False):
            root = self.CRU(root, training=training)

            if FLAGS.attention:
                size = root.get_shape().as_list()
                att = tf.nn.avg_pool(root, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                     padding='SAME')
                att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,
                                        name='conv_att')
                root = tf.multiply(root, att)

        with tf.variable_scope('CRU/3', reuse=False):
            root = self.CRU(root, training=training)

            if FLAGS.attention:
                size = root.get_shape().as_list()
                att = tf.nn.avg_pool(root, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                     padding='SAME')
                att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,
                                        name='conv_att')
                root = tf.multiply(root, att)

        with tf.variable_scope('CRU/4'):
            feature = self.CRU(root, max_pool=False, training=training)

            if FLAGS.attention:
                size = feature.get_shape().as_list()
                att = tf.nn.avg_pool(feature, ksize=[1, size[1], size[2], 1], strides=[1, size[1], size[2], 1],
                                     padding='SAME')
                att = tf.layers.conv2d(att, size[-1], 1, padding='same', kernel_initializer=init,
                                        name='conv_att')
                feature = tf.multiply(feature, att)

        with tf.variable_scope('CRU/5'):
            binary, facial_mask = self.SFL(feature)

        return binary, facial_mask

    def CRU(self, input, max_pool=True, training=True):
        conv1 = self.conv(input, filters=FLAGS.base_num_filters, kernel_size=3, padding='same', use_bias=False,
                                 name='conv1')
        conv1 = tf.layers.batch_normalization(conv1, training=training, name='bn1')
        conv1 = self.active(conv1)

        conv2 = self.conv(conv1, filters=FLAGS.base_num_filters, kernel_size=3, padding='same', use_bias=False,
                                 name='conv2')
        conv2 = tf.layers.batch_normalization(conv2, training=training, name='bn2')
        conv2 = input + conv2

        conv2 = self.active(conv2)

        conv3 = self.conv(conv2, filters=FLAGS.base_num_filters, kernel_size=3, padding='same', use_bias=False,
                                 name='conv3')
        conv3 = tf.layers.batch_normalization(conv3, training=training, name='bn3')
        conv3 = self.active(conv3)

        conv4 = self.conv(conv3, filters=FLAGS.base_num_filters, kernel_size=3, padding='same', use_bias=False,
                                 name='conv4')
        conv4 = tf.layers.batch_normalization(conv4, training=training, name='bn4')

        out = conv2 + conv4
        out = self.active(out)

        if max_pool:
            out = tf.layers.max_pooling2d(out, pool_size=[3, 3], strides=[2, 2], padding='same')

        return out

    def SFL(self, input, training=True):
        mask = self.conv(input, filters=2, kernel_size=3, padding='same', use_bias=False, name='mask_conv')
        mask = tf.nn.sigmoid(mask)

        binary = self.conv(input, filters=FLAGS.base_num_filters, kernel_size=3, strides=2,
                                  padding='same', use_bias=False, name='binary_conv1')
        binary = tf.layers.batch_normalization(binary, training=training, name='bn1')
        binary = self.active(binary)

        binary = self.conv(binary, filters=FLAGS.base_num_filters, kernel_size=3, strides=2,
                                  padding='same', use_bias=False, name='binary_conv2')
        binary = tf.layers.batch_normalization(binary, training=training, name='bn2')
        binary = self.active(binary)

        binary = self.conv(binary, filters=FLAGS.base_num_filters * 2, kernel_size=3, strides=2,
                                  padding='same', use_bias=False, name='binary_conv3')
        binary = tf.layers.batch_normalization(binary, training=training, name='bn3')
        binary = self.active(binary)

        binary = self.conv(binary, filters=FLAGS.base_num_filters * 4, kernel_size=4, strides=2,
                                  padding='valid', use_bias=False, name='binary_conv4')
        binary = tf.layers.batch_normalization(binary, training=training, name='bn4')
        binary = self.active(binary)

        binary = tf.layers.flatten(binary)

        binary = tf.layers.dense(binary, units=256, use_bias=False, name='binary_fc1')
        binary = tf.layers.batch_normalization(binary, training=training, name='bn5')
        binary = self.active(binary)
        binary = tf.layers.dense(binary, units=2, use_bias=False, name='binary_fc2')

        return mask, binary