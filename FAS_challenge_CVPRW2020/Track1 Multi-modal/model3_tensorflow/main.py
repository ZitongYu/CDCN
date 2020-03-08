import numpy as np
import tensorflow as tf
import datetime
import time

from model import Model, get_err_threhold, performances
from tensorflow.python.platform import flags
from data import DataSet

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iterations', 90000, 'number of training iterations.')

# Training options
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 0.001, 'the base learning rate')
flags.DEFINE_integer('lr_decay_itr', 0, 'number of iteration that the lr decays')

flags.DEFINE_float('l2_alpha', 0.00001, 'param of the l2_norm loss')
flags.DEFINE_float('l1_alpha', 0.001, 'param of the l1_norm loss')
flags.DEFINE_float('dropout', 0.1, 'param of the l1_norm loss')
flags.DEFINE_float('loss_alpha1', 1, 'param of the l1_norm loss')
flags.DEFINE_float('loss_alpha2', 1, 'param of the l1_norm loss')
flags.DEFINE_float('score_alpha', 0.5, 'param of the l1_norm loss')

flags.DEFINE_bool('attention', True, 'param of the l1_norm loss')
flags.DEFINE_bool('leaky_relu', True, 'param of the l1_norm loss')
flags.DEFINE_bool('CDC', True, 'param of the l1_norm loss')
flags.DEFINE_string('network', 'DTN', 'network name')
flags.DEFINE_integer('base_num_filters', 8, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('last_relu', True, 'whether to use bias in the attention operation')
flags.DEFINE_bool('last_bn', True, 'whether to use bias in the attention operation')
# flags.DEFINE_bool('clahe', True, 'whether to use bias in the attention operation')
flags.DEFINE_string('loss', 'L2', 'L2 or Con')
flags.DEFINE_bool('bn_nn', False, '')
flags.DEFINE_integer('num_gpus', 1, '')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/miniimagenet1shot/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', 300, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('net', False, 'whether use the data saved on the risk disk, or use the data saved on the local disk.')

flags.DEFINE_integer('protocol', 3, '')


def train(model, saver, sess, exp_string, dataset, resume_itr=0):
    SUMMARY_INTERVAL = 100

    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = 100

    min_ACER_itr = 0
    min_ACER = 1

    print('Done initializing, starting training.')
    print(exp_string)
    losses_map, losses_binary = [], []

    for itr in range(resume_itr, FLAGS.train_iterations):
        # 调节learning rate
        if FLAGS.lr_decay_itr > 0:
            lr = FLAGS.lr * 0.5 ** int(itr / FLAGS.lr_decay_itr)

            if int(itr % FLAGS.lr_decay_itr) < 2:
                print('change the mata lr to:' + str(lr) + ', ----------------------------')
        else:
            lr = FLAGS.lr

        feed_dict = {model.lr: lr}
        feed_dict_data = {}

        if itr == resume_itr:
            image_labels = dataset.get_train_data(FLAGS.batch_size)
            [files, labels] = zip(*image_labels)
            feed_dict_data[dataset.image_lists] = files
            sess.run(dataset.iterator, feed_dict=feed_dict_data)
            faces, depthes, IRs = sess.run(dataset.out_images)
            lbls = np.array(labels)

        feed_dict[model.face] = faces
        feed_dict[model.depth] = depthes
        feed_dict[model.IR] = IRs
        feed_dict[model.label] = lbls
        feed_dict[model.train_flag] = True

        image_labels = dataset.get_train_data(FLAGS.batch_size)
        [files, labels] = zip(*image_labels)
        feed_dict_data[dataset.image_lists] = files
        lbls = np.array(labels)
        sess.run(dataset.iterator, feed_dict=feed_dict_data)

        input_tensors = [model.train_op, model.loss_map, model.loss_binary, model.score_map, model.score_binary, dataset.out_images]

        result = sess.run(input_tensors, feed_dict)

        losses_map.append(result[1])
        losses_binary.append(result[2])

        faces, depthes, IRs = result[-1]

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(losses_map)) + ',   ' + str(np.mean(losses_binary))
            print(str(datetime.datetime.now())[:-7], print_str)
            losses_map, losses_binary = [], []

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            val_labels, val_scores_map, val_scores_binary, val_losses_map, val_losses_binary = [], [], [], [], []
            val_end = False

            first_test_itr = True
            while not val_end:
                feed_dict_test = {model.lr: 0}
                feed_dict_test_data = {}
                if first_test_itr:
                    first_test_itr = False

                    val_image_labels, val_end = dataset.get_val_data(FLAGS.batch_size*2, val=True)
                    [val_files, val_label] = zip(*val_image_labels)
                    val_lbls = np.array(val_label)
                    feed_dict_test_data[dataset.image_lists_val] = val_files
                    sess.run(dataset.iterator_val, feed_dict=feed_dict_test_data)
                    val_faces, val_depthes, val_IRs = sess.run(dataset.out_images_val)

                feed_dict_test[model.face] = val_faces
                feed_dict_test[model.depth] = val_depthes
                feed_dict_test[model.IR] = val_IRs
                feed_dict_test[model.label] = val_lbls
                feed_dict_test[model.train_flag] = False
                val_labels.append(np.mean(val_lbls))

                val_image_labels, val_end = dataset.get_val_data(FLAGS.batch_size * 2, val=True)
                [val_files, val_label] = zip(*val_image_labels)
                val_lbls = np.array(val_label)
                feed_dict_test_data[dataset.image_lists_val] = val_files
                sess.run(dataset.iterator_val, feed_dict=feed_dict_test_data)

                input_tensors = [model.loss_map, model.loss_binary, model.score_map, model.score_binary, dataset.out_images_val]

                result = sess.run(input_tensors, feed_dict_test)

                val_losses_map.append(result[0])
                val_losses_binary.append(result[1])
                val_scores_map.append(np.mean(result[2]))
                val_scores_binary.append(np.mean(result[3]))


                val_faces, val_depthes, val_IRs = result[-1]

                if val_end:
                    feed_dict_test[model.face] = val_faces
                    feed_dict_test[model.depth] = val_depthes
                    feed_dict_test[model.IR] = val_IRs
                    feed_dict_test[model.label] = val_lbls
                    feed_dict_test[model.train_flag] = False
                    val_labels.append(np.mean(val_lbls))
                    input_tensors = [model.loss_map, model.loss_binary, model.score_map, model.score_binary]

                    result = sess.run(input_tensors, feed_dict_test)

                    val_losses_map.append(result[0])
                    val_losses_binary.append(result[1])
                    val_scores_map.append(np.mean(result[2]))
                    val_scores_binary.append(np.mean(result[3]))
                    # print(result[0], result[1], result[2], result[3])
                    #print('----------------------------------------')


            # labels = np.concatenate(labels, axis=0)
            # labels = labels>0
            # labels = labels.astype(np.int32)
            # scores = np.concatenate(scores, axis=0)
            print('----------------------------------------', itr)
            val_scores_map = np.array(val_scores_map)
            val_scores_binary = np.array(val_scores_binary)
            val_scores = FLAGS.score_alpha * val_scores_map + (1-FLAGS.score_alpha) * val_scores_binary

            val_losses_map = np.array(val_losses_map)
            val_losses_binary = np.array(val_losses_binary)
            val_losses = FLAGS.loss_alpha1 * val_losses_map + FLAGS.loss_alpha2 * val_losses_binary

            val_labels = np.array(val_labels)
            accs = performances(val_scores, val_labels)
            APCER = accs[3]
            NPCER = accs[5]
            TPR = accs[6]
            TNR = accs[4]
            ACER = accs[-3]
            print('Total : TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:', np.mean(val_losses))
            if ACER < min_ACER:
                min_ACER_itr = itr
                min_ACER = ACER

            accs = performances(val_scores_map, val_labels)
            APCER = accs[3]
            NPCER = accs[5]
            TPR = accs[6]
            TNR = accs[4]
            ACER = accs[-3]
            print('Map   : TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:', np.mean(val_losses_map))

            accs = performances(val_scores_binary, val_labels)
            APCER = accs[3]
            NPCER = accs[5]
            TPR = accs[6]
            TNR = accs[4]
            ACER = accs[-3]
            print('Binary: TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:', np.mean(val_losses_binary))

            print('min_acer:', min_ACER, 'min_itr:', min_ACER_itr)
            print('----------------------------------------', )

            saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))


def test(model, sess, dataset):
    val_video_score_map, val_video_score_binary = {}, {}
    labels, scores_map, scores_binary, losses_map, losses_binary = [], [], [], [], []
    val_end = False

    while not val_end:
        feed_dict_test = {model.lr: 0}
        feed_dict_test_data = {}

        val_image_labels, val_end = dataset.get_val_data(FLAGS.batch_size*2, val=True)
        [val_files, val_labels] = zip(*val_image_labels)
        val_lbls = np.array(val_labels)
        video = val_files[0]
        index1 = video.find('CASIA_SURF_CeFA')
        index2 = video.find('profile')
        video_name = video[index1+16:index2-1]
        feed_dict_test_data[dataset.image_lists_val] = val_files
        sess.run(dataset.iterator_val, feed_dict=feed_dict_test_data)
        val_faces, val_depthes, val_IRs = sess.run(dataset.out_images_val)

        feed_dict_test[model.face] = val_faces
        feed_dict_test[model.depth] = val_depthes
        feed_dict_test[model.IR] = val_IRs
        feed_dict_test[model.label] = val_lbls
        feed_dict_test[model.train_flag] = False
        labels.append(np.mean(val_lbls))

        input_tensors = [model.loss_map, model.loss_binary, model.score_map, model.score_binary]

        result = sess.run(input_tensors, feed_dict_test)

        score_map = np.mean(result[2])
        score_binary = np.mean(result[3])
        val_video_score_map[video_name] = score_map
        val_video_score_binary[video_name] = score_binary
        scores_map.append(score_map)
        scores_binary.append(score_binary)
        losses_map.append(result[0])
        losses_binary.append(result[1])

    print('-----------------------------------------', )
    scores_map = np.array(scores_map)
    scores_binary = np.array(scores_binary)
    scores = FLAGS.score_alpha * scores_map + (1 - FLAGS.score_alpha) * scores_binary

    losses_map = np.array(losses_map)
    losses_binary = np.array(losses_binary)
    losses = FLAGS.loss_alpha1 * losses_map + FLAGS.loss_alpha2 * losses_binary

    labels = np.array(labels)
    accs = performances(scores, labels)
    APCER = accs[3]
    NPCER = accs[5]
    TPR = accs[6]
    TNR = accs[4]
    ACER = accs[-3]
    print('Total :Val result: TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:',
          np.mean(losses))

    accs = performances(scores_map, labels)
    APCER = accs[3]
    NPCER = accs[5]
    TPR = accs[6]
    TNR = accs[4]
    ACER = accs[-3]
    print('Map   :Val result: TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:',
          np.mean(losses_map))

    accs = performances(scores_binary, labels)
    APCER = accs[3]
    NPCER = accs[5]
    TPR = accs[6]
    TNR = accs[4]
    ACER = accs[-3]
    print('Binary:Val result: TPR:', TPR, 'TNR:', TNR, 'APCER:', APCER, 'NPCER:', NPCER, 'acer:', ACER, 'loss:',
          np.mean(losses_binary))


    test_video_score_map, test_video_score_binary = {}, {}
    val_end = False

    while not val_end:
        feed_dict_test = {model.lr: 0}
        feed_dict_test_data = {}

        val_image_labels, val_end = dataset.get_val_data(FLAGS.batch_size * 2, val=False)
        val_files = val_image_labels
        video = val_files[0]
        index1 = video.find('CASIA_SURF_CeFA')
        index2 = video.find('profile')
        video_name = video[index1 + 16:index2 - 1]
        feed_dict_test_data[dataset.image_lists] = val_files
        sess.run(dataset.iterator, feed_dict=feed_dict_test_data)
        val_faces, val_depthes, val_IRs = sess.run(dataset.out_images)

        feed_dict_test[model.face] = val_faces
        feed_dict_test[model.depth] = val_depthes
        feed_dict_test[model.IR] = val_IRs
        feed_dict_test[model.train_flag] = False

        input_tensors = [model.score_map, model.score_binary]

        result = sess.run(input_tensors, feed_dict_test)

        score_map = np.mean(result[0])
        score_binary = np.mean(result[1])
        test_video_score_map[video_name] = score_map
        test_video_score_binary[video_name] = score_binary

    root = '/media/qyx/34617F8B75F5C8ED/BaiduNetdiskDownload/CASIA_SURF_CeFA'
    val_res = open(root + '/4@' + str(FLAGS.protocol) + '_dev_res.txt', 'r')
    lines = val_res.readlines()
    val_res.close()
    lines_total, lines_map, lines_binary = [], [], []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        score = FLAGS.score_alpha * val_video_score_map[line] + (1 - FLAGS.score_alpha) * val_video_score_binary[line]
        lines_total.append(line + ' ' + str(score)  + '\n')
        lines_map.append(line + ' ' + str(val_video_score_map[line]) + '\n')
        lines_binary.append(line + ' ' + str(val_video_score_binary[line]) + '\n')

    save_root = 'result'
    val_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_dev_res_total.txt', 'w')
    val_res.writelines(lines_total)
    val_res.close()

    val_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_dev_res_map.txt', 'w')
    val_res.writelines(lines_map)
    val_res.close()

    val_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_dev_res_binary.txt', 'w')
    val_res.writelines(lines_binary)
    val_res.close()

    test_res = open(root + '/4@' + str(FLAGS.protocol) + '_test_res.txt', 'r')
    lines = test_res.readlines()
    test_res.close()
    lines_total, lines_map, lines_binary = [], [], []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        score = FLAGS.score_alpha * test_video_score_map[line] + (1 - FLAGS.score_alpha) * test_video_score_binary[line]
        lines_total.append(line + ' ' + str(score) + '\n')
        lines_map.append(line + ' ' + str(test_video_score_map[line]) + '\n')
        lines_binary.append(line + ' ' + str(test_video_score_binary[line]) + '\n')

    test_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_test_res_total.txt', 'w')
    test_res.writelines(lines_total)
    test_res.close()

    test_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_test_res_map.txt', 'w')
    test_res.writelines(lines_map)
    test_res.close()

    test_res = open(save_root + '/4@' + str(FLAGS.protocol) + '_test_res_binary.txt', 'w')
    test_res.writelines(lines_binary)
    test_res.close()

    print('done!')



def main():
    FLAGS.logdir = 'logs/FAS/'

    print('preparing data')
    dataset = DataSet()   #define the task generator

    print('initializing the model')
    model = Model()
    model.create_model()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)

    sess = tf.InteractiveSession()

    exp_string = str(FLAGS.network) +'.bs_'+str(FLAGS.batch_size) + '.lr_' + str(FLAGS.lr)
    exp_string += '.l1_' + str(FLAGS.l1_alpha) +'.l2_' + str(FLAGS.l2_alpha)
    exp_string += '.dr_' + str(FLAGS.dropout) + '.la_' + str(FLAGS.loss_alpha1) + '_' + str(FLAGS.loss_alpha2)
    exp_string += '.nfs_' + str(FLAGS.base_num_filters)
    exp_string += '.L_' + str(FLAGS.loss)

    if FLAGS.lr_decay_itr > 0:
        exp_string += '.decay' + str(FLAGS.lr_decay_itr/1000)
    # if FLAGS.bn_nn:
    #     exp_string += '.bn1'
    # else:
    #     exp_string += '.bn2'
    if FLAGS.attention: exp_string+='.A'
    if FLAGS.leaky_relu: exp_string+='.K'
    if FLAGS.CDC: exp_string+='.C'
    if FLAGS.last_relu: exp_string += '.R'
    if FLAGS.last_bn: exp_string += '.B'

    exp_string += '.' + str(FLAGS.protocol)

    path = FLAGS.logdir + exp_string
    import os
    if not FLAGS.train:
        root_path = os.path.abspath('.')
        if not os.path.exists(os.path.join(root_path, path)):
            exp_string = exp_string[:-3]

    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, dataset, resume_itr)
    else:
        model_file = FLAGS.logdir + '/' + exp_string + '/model' + str(FLAGS.test_iter)
        #model_file = 'module/model2000'
        print("Restoring model weights from " + model_file)
        saver.restore(sess, model_file)
        test(model, sess, dataset)



if __name__ == "__main__":
    main()





