
# coding: utf-8

# In[22]:

import tensorflow as tf
import pickle
import numpy as np
import random
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate, Card, Deck
import pandas as pd

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())

def gen_card_im(card):
    a = np.zeros((4, 13))
    s = suits.index(card.suit)
    r = ranks.index(card.rank)
    a[s, r] = 1
    return np.pad(a, ((6, 7), (2, 2)), 'constant', constant_values=0)

def process_img(img):
    return np.reshape(img, [17 * 17 * 1])

def img_from_cards(cards):
    imgs = np.zeros((len(cards), 17, 17))
    for i, c in enumerate(cards):
        imgs[i] = gen_card_im(Card.from_str(c))
    return imgs.sum(axis=0)

def get_triple(cards):
    if len(cards) == 2:
        my = img_from_cards(cards)
        com = np.zeros((17, 17))
        un = my.copy()
    else:
        my = img_from_cards(cards[:2])
        com = img_from_cards(cards[2:])
        un = img_from_cards(cards)
    return my, com, un

class ConvNet():
    def __init__(self, embedding_size=32):
        self.embedding_size = embedding_size
        self.trainer = tf.train.AdamOptimizer()
        
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.input1 = tf.placeholder(tf.float32, [None, 17, 17, 1])
        self.input2 = tf.placeholder(tf.float32, [None, 17, 17, 1])
        self.input3 = tf.placeholder(tf.float32, [None, 17, 17, 1])

        xavier_init = tf.contrib.layers.xavier_initializer()

        self.conv1_i1 = tf.layers.conv2d(self.input1, 64, 5, 2, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=None, name='conv1')
        self.conv1_i2 = tf.layers.conv2d(self.input2, 64, 5, 2, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv1')
        self.conv1_i3 = tf.layers.conv2d(self.input3, 64, 5, 2, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv1')
        
        self.conv2_i1 = tf.layers.conv2d(self.conv1_i1, 32, 3, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=None, name='conv2')
        self.conv2_i2 = tf.layers.conv2d(self.conv1_i2, 32, 3, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv2')
        self.conv2_i3 = tf.layers.conv2d(self.conv1_i3, 32, 3, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv2')
        
        self.conv3_i1 = tf.layers.conv2d(self.conv2_i1, self.embedding_size, 5, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=None, name='conv3')
        self.conv3_i2 = tf.layers.conv2d(self.conv2_i2, self.embedding_size, 5, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv3')
        self.conv3_i3 = tf.layers.conv2d(self.conv2_i3, self.embedding_size, 5, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init, reuse=True, name='conv3')
        
        self.conv3_flat_i1 = tf.contrib.layers.flatten(self.conv3_i1)
        self.conv3_flat_i2 = tf.contrib.layers.flatten(self.conv3_i2)
        self.conv3_flat_i3 = tf.contrib.layers.flatten(self.conv3_i3)
        
        self.merge = tf.concat([self.conv3_flat_i1, self.conv3_flat_i2, self.conv3_flat_i3], axis=1)
        self.out = tf.layers.dense(self.merge, 1, activation=tf.nn.sigmoid, kernel_initializer=xavier_init)
        self.cross_entropy = tf.reduce_mean(-self.y * tf.log(tf.maximum(self.out, 0.00001))
                                            - (1 - self.y) * tf.log(tf.maximum(1 - self.out, 0.00001)))  
        self.update = self.trainer.minimize(self.cross_entropy)
        self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.out, self.y)))
        
        variables = tf.trainable_variables()
        self.var_norms = tf.global_norm(variables)
#         self.summary_writer = tf.summary.FileWriter('../log/ConvNet/')

def make_features_for_train(df, model_path='../cache/models/ConvNet/'):
    '''Making features with embeddings of cards for all dataset by batches'''
    # df = pd.read_csv('../../../../Downloads/table1_17.csv.zip', header=None)
    # cards = df.iloc[:, 8:].values
    n = max(1, df.shape[0] // 128)
    tf.reset_default_graph()
    net = ConvNet()
    saver = tf.train.Saver()
    cards = df.values
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_path)
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        a_all, b_all, c_all, win_rate_all = None, None, None, None

        for i in range(n):
            if i == n - 1:
                batch = cards[i * 128:]
            else:
                batch = cards[i * 128: (i + 1) * 128]

            if i % 500 == 0:
                print(i)

            mys, coms, uns, ys = [], [], [], []
            for k in range(len(batch)):
                b = [x for x in batch[k] if x is not np.nan]
                my, com, un = get_triple(b)
                mys.append(np.expand_dims(np.expand_dims(my, 0), -1))
                coms.append(np.expand_dims(np.expand_dims(com, 0), -1))
                uns.append(np.expand_dims(np.expand_dims(un, 0), -1))
            mys = np.vstack(mys)
            coms = np.vstack(coms)
            uns = np.vstack(uns)

            feed_dict = {
                net.input1: mys,
                net.input2: coms,
                net.input3: uns
            }
            a, b, c, win_rate = sess.run([net.conv3_flat_i1, net.conv3_flat_i2, net.conv3_flat_i3,
                                          net.out], feed_dict=feed_dict)  
            if a_all is not None:
                a_all = np.vstack((a_all, a))
                b_all = np.vstack((b_all, b))
                c_all = np.vstack((c_all, c))
                win_rate_all = np.vstack((win_rate_all, win_rate))
            else:
                a_all = a
                b_all = b
                c_all = c
                win_rate_all = win_rate
    features = np.hstack((a_all, b_all, c_all, win_rate_all))
    return features

sess = None
net = None

def init_model(model_path='../cache/models/ConvNet/'):
    '''Initialize tf model for evaluation'''
    global sess, net
    tf.reset_default_graph()
    net = ConvNet()
    saver = tf.train.Saver()
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    
def make_features_for_one(cards):
    '''Making features with embeddings of cards for one example'''
    # df = pd.read_csv('../../../../Downloads/table1_17.csv.zip', header=None)
    if sess is None or net is None:
        raise Exception('You must call init_model() first!')

    with sess.as_default(), sess.graph.as_default():
        my, com, un = get_triple(cards)
        my = np.expand_dims(np.expand_dims(my, 0), -1)
        com = np.expand_dims(np.expand_dims(com, 0), -1)
        un = np.expand_dims(np.expand_dims(un, 0), -1)

        feed_dict = {
            net.input1: my,
            net.input2: com,
            net.input3: un
        }
        a, b, c, win_rate = sess.run([net.conv3_flat_i1, net.conv3_flat_i2, net.conv3_flat_i3,
                                      net.out], feed_dict=feed_dict)
        a = a[0]
        b = b[0]
        c = c[0]
        win_rate = win_rate[0]

        features = np.hstack((a, b, c, win_rate))
        return features
    
# Examples

# df = pd.DataFrame([['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
#                    ['D2', 'D3', 'D4', 'D6', 'D7', 'D8', 'CA'],
#                    ['D2', 'D3', np.nan, np.nan,np.nan,np.nan,np.nan],
#                    ['D2', 'C2', np.nan, np.nan,np.nan,np.nan,np.nan]])
# fs = make_features_for_train(df)

# init_model()
# f = make_features_for_one(['D2', 'D3'])
# f = make_features_for_one(['D2', 'C2'])

