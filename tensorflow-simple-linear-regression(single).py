import tensorflow as tf
import matplotlib.pyplot as plt


X_data = [10, 8, 3, 2, 12, 5, 4, 3, 1]
Y_data = [90, 80, 50, 30, 100, 60, 45, 40, 10]

X = tf.placeholder(dtype=tf.float32, name='Input')
Y = tf.placeholder(dtype=tf.float32, name='output')

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
trainer = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_graph = []
print('start learning')
for i in range(1000):
    _, cost_val = sess.run(fetches=[trainer, cost], feed_dict={X: X_data, Y: Y_data})
    cost_graph.append(cost_val)
    print('step = ', i, ', cost_val = ', cost_val, '\n')
print('end learning')

test_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
result_data = sess.run(fetches=[hypothesis], feed_dict={X:test_time})
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(X_data, Y_data, 'bo')
plt.plot(test_time, result_data[0], 'green')
plt.ylabel('score')
plt.legend(['data', 'hypothesis'], loc='upper left')
plt.subplot(122)
plt.ylabel('cost')
plt.plot(cost_graph)
plt.savefig('./linear-regression/pre-trained-linear-regression-single.png')
plt.show()

