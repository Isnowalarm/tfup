# -*- coding=utf-8 -*-
"""
#
# File Name: rnn_model.py
# @Author: lidonghui02@baidu.com
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
import os
import numpy as np
import time

class CharRNN(object):
    def __init__(self,num_classes,num_seqs = 64,num_steps = 50,lstm_size = 128,num_layers = 2,
            learning_rate = 0.001, grad_clip = 5, sampling = False, train_keep_prob = 0.5,
            use_embedding = False, embedding_size = 128):
    
        if sampling:
            num_seqs, num_steps = 1, 1
        
        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        
        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
    
    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32,shape = (self.num_seqs,self.num_steps), name = 'inputs')
            self.targets = tf.placeholder(tf.int32,shape = (self.num_seqs,self.num_steps), name = 'targets')
            self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

            if self.use_embedding:
                with tf.device('/cpu:0'):
                    embedding = tf.get_variable('embedding',[self.num_classes,self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding,self.inputs)
            else:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)


    def build_lstm(self):
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob = keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size,self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = cell.zero_state(self.num_seqs,tf.float32) 
            #print self.initial_state

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state = self.initial_state)
            #print self.lstm_outputs
            seq_output = tf.concat(self.lstm_outputs,1)
            #print seq_output     
            x = tf.reshape(seq_output, [-1, self.lstm_size])
            #print x 
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes],stddev = 0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            #print self.logits
            self.proba_prediction = tf.nn.softmax(self.logits, name = 'predictions')
            #print self.proba_prediction

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets,self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads,tvars))

   
    def train(self, batch_generator,max_steps,save_path,save_every_n,log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob:self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer], feed_dict = feed)
                end = time.time()
            
                if step % log_every_n == 0:
                    print 'step: {}/{}...'.format(step, max_steps) 
                    print 'loss: {:.4f}...'.format(batch_loss)
                    print '{:.4f} sec/batch'.format(end - start)
                if step % save_every_n == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step = step)

                if step >= max_steps:
                    break

    def sample(self, n_samples, prime, vacab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))
        for c in prime:
            x = np.zeros((1,1))
            x[0,0]  = c
            feed = {self.inputs: x,
                   self.keep_prob:1,
                   self.initial_state:new_state}
            preds, new_state = sess.run([self.proba_prediction,self.final_state], feed_dict = feed)
        c = pick_top_n(preds, vocab_size)
        samples.append(c)

        for i in range(n_samples):
            x = np.zeros((1,1))
            x[0,0]  = c
            feed = {self.inputs: x,
                   self.keep_prob:1,
                   self.initial_state:new_state}
            preds, new_state = sess.run([self.proba_prediction,self.final_state], feed_dict = feed)
            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(c)
    
    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print 'Restore from : {}'.format(checkpoint)

if __name__ == '__main__':
    cr = CharRNN(100)
