import tensorflow as tf
import matplotlib.pyplot as plt


X_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

Y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
trainer = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_graph = []

print('start learning')
for i in range(10000):
    _, cost_val = sess.run(fetches=[trainer, cost], feed_dict={X: X_data, Y: Y_data})
    cost_graph.append(cost_val)
    if cost_val < 3:
        break
    print('step = ', i, ', cost_val = ', cost_val, '\n')
print('end learning')

test_time = [[80., 55., 30.]]
result_data = sess.run(fetches=[hypothesis], feed_dict={X:test_time})
print(result_data[0])
plt.ylabel('cost')
plt.plot(cost_graph)
plt.savefig('./linear-regression/pre-trained-linear-regression-multivariable.png')
plt.show()

