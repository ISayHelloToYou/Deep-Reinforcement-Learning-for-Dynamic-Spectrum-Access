import tensorflow as tf
import numpy as np
# tf.reset_default_graph()
# b3 = tf.Variable(np.arange(2).reshape((2,)), dtype = tf.float32, name = 'biases3')
# saver = tf.train.Saver()
# with tf.variable_scope('main/main'):
#     w3 = tf.Variable(tf.random_uniform([128, 2]), dtype=tf.float32, name='weights3')
#     # w3 = tf.Variable(np.arange(128*2).reshape((128,2)), dtype=tf.float32, name='weights3')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('/home/fengzhike/PycharmProjects/ceshi/ceshi_checkpoints/dqn_multi-user.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/fengzhike/PycharmProjects/ceshi/ceshi_checkpoints'))
    # saver.restore(sess, '/home/fengzhike/PycharmProjects/ceshi/ceshi_checkpoints/dqn_multi-user.ckpt')
    print(sess.run('weights3:0'))



# from tensorflow.contrib.framework.python.framework import checkpoint_utils
# var_list = checkpoint_utils.list_variables("ceshi_checkpoints/dqn_multi-user.ckpt")
# for v in var_list:
#     print(v)
