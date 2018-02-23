import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(0.2)
biases = tf.Variable(0.0)

y = Weights * x_data + biases

# tensorflow 结构
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

#  初始化参数
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(Weights), sess.run(biases))
for step in range(200):
    sess.run(train)

print(sess.run(Weights), sess.run(biases))
sess.close()
