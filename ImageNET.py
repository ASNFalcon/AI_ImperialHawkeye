import tensorflow as tf

a= tf.constant([1.0, 2.0], name = "a")
b= tf.constant([4.0, 3.0], name = "b")

result = tf.add(a, b, name = "add")

sess = tf.Session()
sess.run(result)

print(result)
