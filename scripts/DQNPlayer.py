from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import Card, Deck
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.api.emulator import Emulator, ActionChecker, RoundManager,MessageBuilder, Const, Event, DataEncoder
import pickle
import tensorflow as tf

import sys
sys.path.insert(0, '../scripts/')
from util import *


class DQNPlayer(BasePokerPlayer):
    '''
    DQN Player, bot wich use Double-Dueling-DQN architecture.

    Parametrs
    ---------
    h_size : shape of layer after conv part (also before double part too)

    lr : learning rate of the optimizer

    gradient_clip_norm : gradients of the loss function will be clipped by this value
    
    total_num_actions : the number of actions witch agent can choose

    is_double : whether or not to use the double architecture

    is_main : whether or not to use this agent as main (when using the dueling architecture)

    is_restore : wheter or not to use pretrained weight of the network

    is_train : whether or not to use this agent for training

    is_debug  wheter or not to print the debug information
    '''
        
    def __init__(self, h_size=128, lr=0.0001, gradient_clip_norm=500, total_num_actions=5, is_double=False,
                 is_main=True, is_restore=False, is_train=True, debug=False):              
        self.h_size = h_size
        self.lr = lr
        self.gradient_clip_norm = gradient_clip_norm
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_restore = is_restore
        self.is_train = is_train
        self.debug = debug
        
        with open('../cache/hole_card_estimation.pkl', 'rb') as f:
            self.hole_card_est = pickle.load(f)
        
        if not is_train:
            tf.reset_default_graph()
        
        self.scalar_input = tf.placeholder(tf.float32, [None, 17 * 17 * 1])
        self.features_input = tf.placeholder(tf.float32, [None, 20])
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.img_in = tf.reshape(self.scalar_input, [-1, 17, 17, 1])
        self.conv1 = tf.layers.conv2d(self.img_in, 32, 5, 2, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 3, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.conv3 = tf.layers.conv2d(self.conv2, self.h_size, 5, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init)
        self.conv3_flat = tf.contrib.layers.flatten(self.conv3)
        self.conv3_flat = tf.layers.dropout(self.conv3_flat)
        
        self.d1 = tf.layers.dense(self.features_input, 64, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d1 = tf.layers.dropout(self.d1)
        self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d2 = tf.layers.dropout(self.d2)
        
        self.merge = tf.concat([self.conv3_flat, self.d2], axis=1)
        self.d3 = tf.layers.dense(self.merge, 256, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d3 = tf.layers.dropout(self.d3)
        self.d4 = tf.layers.dense(self.d3, self.h_size, activation=tf.nn.elu, kernel_initializer=xavier_init)
        
        if is_double:
            self.stream_A, self.stream_V = tf.split(self.d4, 2, 1)
            self.AW = tf.Variable(xavier_init([self.h_size // 2, total_num_actions]))
            self.VW = tf.Variable(xavier_init([self.h_size // 2, 1]))

            self.advantage = tf.matmul(self.stream_A, self.AW)
            self.value = tf.matmul(self.stream_V, self.VW)

            self.Q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, 1, True))
        else:
            self.Q_out = tf.layers.dense(self.d4, 5, kernel_initializer=xavier_init)
            
        self.predict = tf.argmax(self.Q_out, 1)
        
        self.target_Q = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.int32, [None])
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.Q - self.target_Q)
        self.loss = tf.reduce_mean(self.td_error)
        
        if is_main:
            variables = tf.trainable_variables() # [:len(tf.trainable_variables()) // 2]
            if is_train:
                self._print(len(variables))
                self._print(variables)
            self.gradients = tf.gradients(self.loss, variables)
#             self.grad_norms = tf.global_norm(self.gradients)
            self.var_norms = tf.global_norm(variables)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, gradient_clip_norm)
            self.grad_norms = tf.global_norm(grads)
            self.trainer = tf.train.AdamOptimizer(lr)
#             self.update_model = self.trainer.minimize(self.loss)
            self.update_model = self.trainer.apply_gradients(zip(grads, variables))

            self.summary_writer = tf.summary.FileWriter('../log/DQN/')
            
        if not is_train:
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init)
        
        if is_restore:
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('../cache/models/DQN/')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
    def _print(self, *msg):
        if self.debug:
            print(msg)
        
    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state['street']
        bank = round_state['pot']['main']['amount']
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]
        dealer_btn = round_state['dealer_btn']
        small_blind_pos = round_state['small_blind_pos']
        big_blind_pos = round_state['big_blind_pos']
        next_player = round_state['next_player']
        round_count = round_state['round_count']
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]

        
        self.features = get_street(street)
        self.features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
        self.features.extend(other_stacks)
        self.features.append(estimation)
        
        img_state = img_from_state(hole_card, round_state)
        img_state = process_img(img_state)
        action_num = self.sess.run(self.predict, feed_dict={self.scalar_input: [img_state],
                                                            self.features_input: [self.features]})[0]
        qs = self.sess.run(self.Q_out, feed_dict={self.scalar_input: [img_state],
                                                  self.features_input: [self.features]})[0]
        self._print(qs)
        action, amount = get_action_by_num(action_num, valid_actions)                    

#         if not self.debug and np.random.rand() < 0.2:
#             self.action_num = np.random.randint(0, 5)
        return action, amount
        
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self._print(['Hole:', hole_card])        
        self.start_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]
        self._print(['Start stack:', self.start_stack])
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
        self._print(['Estimation:', estimation])
    
    def receive_street_start_message(self, street, round_state):
        pass
            
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        end_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        self._print(['End stack:', end_stack])

