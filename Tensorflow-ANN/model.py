import numpy as np

import tensorflow as tf


# ----------------------------- #
# ----------------------------- #
class Model(object):


    # ----------------------------- #
    def __init__(self):
        print 'The model is ready! Create the model using the `build()` method.'
        # ----------------------------- #
    

    # ----------------------------- #
    def build(self, x, y):
        self.x = x
        self.y = y
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.setup()
        # ----------------------------- #


    # ----------------------------- #
    def setup(self):
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cross_entropy)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        # ----------------------------- #


    # ----------------------------- #
    def train(self, dataset, num=1000):
        acc_batch = []
        for i in range(num):
            batch_xs, batch_ys = dataset.train.next_batch(32)
            feed_batch = { self.x: batch_xs, self.y_: batch_ys }
            acc, _ = self.sess.run([self.accuracy, self.train_step], feed_dict=feed_batch)
            acc_batch.append(acc)
            if (i+1) % 100 == 0:
                print 'Run {:4d} \tTraining accuracy: {:6.2f}% | {:6.2f}% | {:6.2f}%'.format(
                    i+1, np.mean(acc_batch)*100.0, np.min(acc_batch)*100, np.max(acc_batch)*100)
                acc_batch = []
        # Print accuracy
        feed_test = { self.x: dataset.test.images, self.y_: dataset.test.labels }
        print 'Testing accuracy: {:6.2f}%'.format( self.sess.run(self.accuracy, feed_dict=feed_test) * 100.0)
        # Done
        return self.sess
        # ----------------------------- #


# ----------------------------- #
# ----------------------------- #