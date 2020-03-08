from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import copy
from sklearn.metrics import roc_curve, auc
from scipy import interp


from tensorflow.python.platform import flags
from utils import contrast_depth_loss, L2_loss
from networks import ZZNet, DTN

FLAGS = flags.FLAGS


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1)
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th

def performances(test_scores, test_labels):
    # print('label',test_labels)
    # print('score',test_scores)

    test_labels_bk = copy.deepcopy(test_labels)
    test_scores_bk = copy.deepcopy(test_scores)
    test_labels_bk[test_labels_bk < 0] = 0

    fpr_test, tpr_test, threshold_test = roc_curve(test_labels_bk, test_scores_bk, pos_label=1)
    err, best_th = get_err_threhold(fpr_test, tpr_test, threshold_test)
    precision_th1 = 0.005
    RECALL1 = interp(precision_th1, fpr_test, tpr_test)

    precision_th2 = 0.01
    RECALL2 = interp(precision_th2, fpr_test, tpr_test)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(test_labels.shape[0]):
        if test_labels[i] == 0:
            if test_scores[i] <= best_th:
                TN += 1
            else:
                FP += 1
        else:
            if test_scores[i] > best_th:
                TP += 1
            else:
                FN += 1

    FAR = FP / (FP + TP + 0.000001)  ### False Acceptance Rate
    FRR = FN / (TN + FN + 0.000001)  ### False Rejection Rate
    HTER = (FAR + FRR) / 2  ### Half Total Error Rate
    APCER = FP / (TN + FP + 0.000001)  ### Attack Presentation Classification Error Rate
    TNR = 1 - APCER  ### True Negative Rate
    NPCER = FN / (FN + TP + 0.000001)  ### Normal Presentation Classification Error Rate
    TPR = 1 - NPCER  ### True Positive Rate
    ACER = (APCER + NPCER) / 2  ### Average Classification Error Rate
    ACC = (TP + TN) / (0.000001 + TP + FP + FN + TN)

    return np.float32(FAR), np.float32(FRR), np.float32(HTER), np.float32(APCER), \
           np.float32(TNR), np.float32(NPCER), np.float32(TPR), np.float32(ACER), np.float32(ACC), np.float32(best_th)


class Model:
    def __init__(self, dim_input=1, dim_output=1):
        self.lr = tf.placeholder_with_default(FLAGS.lr, ())
        self.train_flag = tf.placeholder(tf.bool)

        if FLAGS.network=='zznet':
            self.net = ZZNet()
            self.forward = self.net.forward
        elif FLAGS.network=='zznet2':
            self.net = ZZNet()
            self.forward = self.net.forward_split
        elif FLAGS.network=='DTN':
            self.net = DTN()
            self.forward = self.net.forward
        else:
            raise ValueError('Unrecognized network name.')


        if FLAGS.loss == 'L2':
            self.loss_func = L2_loss
        else:
            self.loss_func = contrast_depth_loss

        self.classification = True

        shape = [None, 256, 256, 3]
        self.face = tf.placeholder(tf.float32, shape=shape)
        shape = [None, 256, 256, 1]
        self.depth = tf.placeholder(tf.float32, shape=shape)
        shape = [None, 256, 256, 1]
        self.IR = tf.placeholder(tf.float32, shape=shape)
        shape = [None, ]
        self.label = tf.placeholder(tf.int32, shape=shape)

        self.map = tf.get_variable('map', shape=[1, 32, 32, 1], dtype=tf.int32,
                                   initializer=tf.initializers.ones(dtype=tf.int32), trainable=False)


    def create_model(self):
        faces = tf.split(self.face, axis=0, num_or_size_splits=FLAGS.num_gpus)
        depthes = tf.split(self.depth, axis=0, num_or_size_splits=FLAGS.num_gpus)
        IRs = tf.split(self.IR, axis=0, num_or_size_splits=FLAGS.num_gpus)
        labels = tf.split(self.label, axis=0, num_or_size_splits=FLAGS.num_gpus)

        tower_grads = []
        tower_loss_map = []
        tower_score_map = []
        tower_loss_binary = []
        tower_score_binary = []
        optimizer = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                #with tf.device('/gpu:%d' % i):
                with tf.device('/gpu:0'):
                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        output_map, binary = self.forward(faces[i], depthes[i], IRs[i], training=self.train_flag)
                    label = labels[i]
                    map_label = tf.reshape(label, shape=[-1, 1, 1, 1])
                    map_label = tf.multiply(map_label, self.map)
                    map_label = tf.cast(map_label, dtype=tf.float32)
                    loss_map = self.loss_func(output_map, map_label)
                    score_map = tf.reduce_mean(output_map, axis=[1,2,3])

                    loss_binary = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=binary, labels=label)
                    loss_binary = tf.reduce_mean(loss_binary)
                    score_binary = tf.nn.softmax(binary)[:,1]

                    loss = loss_map*FLAGS.loss_alpha1 + loss_binary*FLAGS.loss_alpha2

                    if 'weights' not in self.__dir__():
                        self.weights = []
                        vars = tf.trainable_variables()
                        for var in vars:
                            if 'conv' in var.name or 'fc' in var.name:
                                self.weights.append(var)
                        # tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv')
                    weight_l_loss0 = 0
                    if FLAGS.l2_alpha > 0:
                        for array in self.weights:
                            weight_l_loss0 += tf.reduce_sum(tf.square(array)) * FLAGS.l2_alpha
                    if FLAGS.l1_alpha > 0:
                        for array in self.weights:
                            weight_l_loss0 += tf.reduce_sum(tf.abs(array)) * FLAGS.l1_alpha

                    gvs = optimizer.compute_gradients(loss + weight_l_loss0, self.weights)
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                    tower_grads.append(gvs)
                    tower_loss_map.append(loss_map)
                    tower_score_map.append(score_map)
                    tower_loss_binary.append(loss_binary)
                    tower_score_binary.append(score_binary)
                    
        loss_map = tf.stack(axis=0, values=tower_loss_map)
        self.loss_map = tf.reduce_mean(loss_map, 0)
        self.score_map = tf.concat(tower_score_map, axis=-1)
        loss_binary = tf.stack(axis=0, values=tower_loss_binary)
        self.loss_binary = tf.reduce_mean(loss_binary, 0)
        self.score_binary = tf.concat(tower_score_binary, axis=-1)
        
        mean_grads = average_gradients(tower_grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        #     train_op = optimizer.apply_gradients(mean_grads)
            self.train_op = optimizer.apply_gradients(mean_grads)



