import tensorflow as tf
import os

a = tf.Variable(tf.constant([1.0, 2.0], name = "a"))
b = tf.Variable(tf.constant([4.0, 3.0], name = "b"))


result = a + b

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, os.path.abspath('../AI_ImperialHawkeye\\model\\model.ckpt'))
    
    sess.run(init_op)
    saver.save(sess, os.path.abspath('../AI_ImperialHawkeye\\model\\model.ckpt'))
