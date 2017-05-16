import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

class vgg16Net:
    def __init__(self, img, weights, sess):
        self.imgs = img
        self.convlayers()
        self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([127.472, 127.876, 127.504], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        # conv3_
        with tf.name_scope('conv3_1') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            filter = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, filter, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [filter, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    # def fc_layers(self):
        # # fc1
        # with tf.name_scope('fc1') as scope:
        #     shape = int(np.prod(self.pool5.get_shape()[1:]))
        #     fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     pool5_flat = tf.reshape(self.pool5, [-1, shape])
        #     fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        #     self.fc1 = tf.nn.relu(fc1l)
        #     self.parameters += [fc1w, fc1b]

        # # fc2
        # with tf.name_scope('fc2') as scope:
        #     fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        #     self.fc2 = tf.nn.relu(fc2l)
        #     self.parameters += [fc2w, fc2b]

        # # fc3
        # with tf.name_scope('fc3') as scope:
        #     fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
        #     self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        keys = keys[:-6]
        for i, k in enumerate(keys):
            # print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def return_layers(self,im,sess):
        # self.load_weights("/vggWeights/vgg16_weights.npz", sess)
        c1 = np.array(im)
        c2,c3,c4,c5 = sess.run((self.conv1_2,self.conv2_2,self.conv3_3,self.conv4_3),feed_dict={self.imgs:im})
        # c2,c3,c4 = sess.run((self.conv1_2,self.conv2_2,self.conv3_3),feed_dict={self.imgs:im})
        c2 = np.array(c2)
        c3 = np.array(c3)
        c4 = np.array(c4)
        c5 = np.array(c5)
        # print(c1.shape,c2.shape,c3.shape,c4.shape)
        # return c1,c2,c3,c4
        return c1,c2,c3,c4,c5

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 256, 256, 3]) #224
    vgg = vgg16Net(imgs, 'vgg16_weights.npz', sess)

    img1 = imread('bea1.jpg', mode='RGB')
    # img1 = imresize(img1, (224, 224))

    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    c1,c2,c3,c4, c5 = vgg.return_layers(img1,sess)
    print(c1.shape, c2.shape, c3.shape, c4.shape, c5.shape)

    # preds = (np.argsort(prob)[::-1])[0:5]