""" Code for loading data. """
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.python.platform import flags
import os
import pwd


FLAGS = flags.FLAGS




def random_float(f_min, f_max):
    return f_min + (f_max-f_min) * random.random()


def Contrast_and_Brightness(img, alpha=None, gamma=None):
    # blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    gamma = random.randint(-40, 40)
    alpha = random_float(0.5, 1.5)
    dst = cv2.addWeighted(img, alpha, img, 0, gamma)
    return dst


def get_cut_out(img, length=50):
    # print(len(img.shape))
    h, w = img.shape[0], img.shape[1]  # Tensor [1][2],  nparray [0][1]
    if len(img.shape) == 3:
        mask = np.ones((h, w, 3), np.float32)
    else:
        mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)

    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    if len(img.shape) == 3:
        mask[y1: y2, x1: x2, :] = 0
    else:
        mask[y1: y2, x1: x2] = 0
    img *= np.array(mask, np.uint8)
    return img


def clip_image(gray):
    im_bool = gray > 1
    im_bool = im_bool.astype(np.uint8)
    a = np.where(im_bool == 1)
    rows = a[0]
    cols = a[1]
    min_row = np.min(rows)
    max_row = np.max(rows)
    min_col = np.min(cols)
    max_col = np.max(cols)

    return min_row, max_row, min_col, max_col


class DataSet(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self):

        if pwd.getpwuid(os.getuid())[0] == 'qyx':
            self.root = '/media/qyx/34617F8B75F5C8ED/BaiduNetdiskDownload/CASIA_SURF_CeFA'
        else:
            self.root = '/home/jdnetusr/qyx/dataset'

        self.train_txt = self.root + '/4@' + str(FLAGS.protocol) + '_train.txt'
        self.val_txt = self.root + '/4@' + str(FLAGS.protocol) + '_dev_ref.txt'
        self.test_txt = self.root + '/4@' + str(FLAGS.protocol) + '_test_res.txt'

        self.train_data_pointer = 0
        self.val_data_pointer = 0
        self.test_data_pointer = 0
        self.prepare_train_data()

        self.image_lists = tf.placeholder(dtype=tf.string, shape=[None, ])
        dataset = tf.data.Dataset.from_tensor_slices(self.image_lists)
        dataset = dataset.map(self.read_image, num_parallel_calls=12)
        dataset = dataset.batch(200)
        iterator = dataset.make_initializable_iterator()
        self.out_images = iterator.get_next()
        self.iterator = iterator.initializer

        self.image_lists_val = tf.placeholder(dtype=tf.string, shape=[None, ])
        dataset_val = tf.data.Dataset.from_tensor_slices(self.image_lists_val)
        dataset_val = dataset_val.map(self.read_image_val, num_parallel_calls=12)
        dataset_val = dataset_val.batch(200)
        iterator_val = dataset_val.make_initializable_iterator()
        self.out_images_val = iterator_val.get_next()
        self.iterator_val = iterator_val.initializer


    def prepare_train_data(self):
        self.train_list = []
        text = open(self.train_txt, 'r')
        contents = text.readlines()
        text.close()
        for content in contents:
            image_label = content.split(' ')
            image_path = image_label[0]
            label = image_label[-1]
            if '\n' in label:
                label = label[:-1]
            label = int(label)
            image_path = os.path.join(self.root, image_path)
            depth_path = image_path.replace('profile', 'depth')
            ir_path = image_path.replace('profile', 'ir')
            if os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(ir_path):
                self.train_list.append([image_path, label])

        self.val_list = []
        text = open(self.val_txt, 'r')
        contents = text.readlines()
        text.close()
        for content in contents:
            video_list = []
            video_label = content.split(' ')
            video_path = video_label[0]
            label = video_label[-1]
            if '\n' in label:
                label = label[:-1]
            video_path = os.path.join(self.root, video_path, 'profile')
            images = os.listdir(video_path)
            for image in images:
                image_path = os.path.join(video_path, image)
                video_list.append([image_path, int(label)])
            self.val_list.append(video_list)

        self.test_list = []
        text = open(self.test_txt, 'r')
        contents = text.readlines()
        text.close()
        for content in contents:
            video_list = []
            if '\n' in content:
                content = content[:-1]
            video_path = os.path.join(self.root, content, 'profile')
            images = os.listdir(video_path)
            for image in images:
                image_path = os.path.join(video_path, image)
                video_list.append(image_path)
            self.test_list.append(video_list)
        
        random.shuffle(self.train_list)
        self.total_train_num = len(self.train_list)
        self.total_val_num = len(self.val_list)
        self.total_test_num = len(self.test_list)
        print(self.total_val_num, self.total_test_num)


    def get_train_data(self, batch_size):
        if self.train_data_pointer + batch_size >= self.total_train_num:
            batch_data = self.train_list[self.train_data_pointer:]
            self.train_data_pointer = 0
            random.shuffle(self.train_list)
        else:
            batch_data = self.train_list[self.train_data_pointer:self.train_data_pointer + batch_size]
            self.train_data_pointer += batch_size

        random.shuffle(batch_data)
        return batch_data


    def get_val_data(self, batch_size, val=True):
        if val:
            if self.val_data_pointer + 1 >= self.total_val_num:
                video = self.val_list[self.val_data_pointer]
                end_itr = True
                self.val_data_pointer = 0
            else:
                video = self.val_list[self.val_data_pointer]
                end_itr = False
                self.val_data_pointer += 1
        else:
            if self.test_data_pointer + 1 >= self.total_test_num:
                video = self.test_list[self.test_data_pointer]
                end_itr = True
                self.test_data_pointer = 0
            else:
                video = self.test_list[self.test_data_pointer]
                end_itr = False
                self.test_data_pointer += 1

        if len(video) >= batch_size:
            batch_data = random.sample(video, batch_size)
        else:
            a = len(video)/FLAGS.num_gpus
            a = int(a)
            batch_data = random.sample(video, a * FLAGS.num_gpus)

        random.shuffle(batch_data)
        return batch_data, end_itr


    def _parser(self, image_path):
        image_path = image_path.decode()
        depth_path = image_path.replace('profile', 'depth')
        ir_path = image_path.replace('profile', 'ir')
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_row, max_row, min_col, max_col = clip_image(gray)
        face = image[min_row:max_row, min_col:max_col, :]

        depth = cv2.imread(depth_path, 0)
        min_row, max_row, min_col, max_col = clip_image(depth)
        face_depth = depth[min_row:max_row, min_col:max_col]

        ir = cv2.imread(ir_path, 0)
        min_row, max_row, min_col, max_col = clip_image(ir)
        face_ir = ir[min_row:max_row, min_col:max_col]

        if random_float(0.0, 1.0) < 0.5:
            is_random_flip = True
        else:
            is_random_flip = False

        if random_float(0.0, 1.0) < 0.5:
            is_change_color = True
        else:
            is_change_color = False

        if random_float(0.0, 1.0) < 0.5:
            is_cut_out = True
        else:
            is_cut_out = False

        if is_random_flip:
            face = cv2.flip(face, 1)
            face_depth = cv2.flip(face_depth, 1)
            face_ir = cv2.flip(face_ir, 1)

        if is_change_color:
            face = Contrast_and_Brightness(face)

        face = cv2.resize(face, (256, 256))
        face_depth = cv2.resize(face_depth, (256, 256))
        face_ir = cv2.resize(face_ir, (256, 256))

        if is_cut_out:
            face = get_cut_out(face)
            face_depth = get_cut_out(face_depth)
            face_ir = get_cut_out(face_ir)

        face = face.astype(np.float32) - 127.5
        face_depth = face_depth.astype(np.float32) - 127.5
        face_ir = face_ir.astype(np.float32) - 127.5
        
        return face, face_depth[:,:, np.newaxis], face_ir[:,:,np.newaxis]


    def read_image(self, face_path):
        face = tf.py_func(self._parser, inp=[face_path], Tout=[tf.float32, tf.float32, tf.float32])
        return face


    def _parser_val(self, image_path):
        image_path = image_path.decode()
        depth_path = image_path.replace('profile', 'depth')
        ir_path = image_path.replace('profile', 'ir')

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_row, max_row, min_col, max_col = clip_image(gray)
        face = image[min_row:max_row, min_col:max_col, :]

        depth = cv2.imread(depth_path, 0)
        min_row, max_row, min_col, max_col = clip_image(depth)
        face_depth = depth[min_row:max_row, min_col:max_col]

        ir = cv2.imread(ir_path, 0)
        min_row, max_row, min_col, max_col = clip_image(ir)
        face_ir = ir[min_row:max_row, min_col:max_col]

        face = cv2.resize(face, (256, 256))
        # print(face.dtype)
        # if FLAGS.clahe:
        #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #     face = clahe.apply(face)
        face_depth = cv2.resize(face_depth, (256, 256))
        face_ir = cv2.resize(face_ir, (256, 256))

        face = face.astype(np.float32) - 127.5
        face_depth = face_depth.astype(np.float32) - 127.5
        face_ir = face_ir.astype(np.float32) - 127.5

        return face, face_depth[:, :, np.newaxis], face_ir[:, :, np.newaxis]


    def read_image_val(self, face_path):
        face = tf.py_func(self._parser_val, inp=[face_path], Tout=[tf.float32, tf.float32, tf.float32])
        return face



if __name__ == "__main__":
    flags.DEFINE_integer('protocol', 1, 'number of metatraining iterations.')

    # Training options
    flags.DEFINE_integer('batch_size', 4, 'number of tasks sampled per meta-update')
    flags.DEFINE_float('lr', 0.001, 'the base learning rate of the generator')
    flags.DEFINE_integer('lr_decay_itr', 0, 'number of iteration that the meta lr should decay')

    data_generator = DataGenerator_dataset()
    for _ in range(2000):
        ims, lbls = data_generator.get_data(20, train=True)
        print(1)
