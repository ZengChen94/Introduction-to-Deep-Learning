import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # call mnist function

learningRate = 1e-3
trainingIters = 50000
batchSize = 100
displayStep = 10

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 128  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0)  # configuring so you can get it as needed for the 28 pixels

    # find which lstm to use in the documentation
    # lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
    # lstmCell = rnn_cell.GRUCell(nHidden)
    # lstmCell = rnn_cell.BasicRNNCell(nHidden)

    # mixed model with RNN and LSTM
    lstmCell = rnn.BasicRNNCell(nHidden)

    outputs, states = rnn.static_rnn(lstmCell, x,
                                     dtype=tf.float32)  # for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

losses_list = []
accs_list = []

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
        loss = sess.run(cost, feed_dict={x: batchX, y: batchY})

        losses_list.append(loss)
        accs_list.append(acc)

        if step % displayStep == 0:
            print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))

    sess.close()

# Plot the accuracy and loss under different parameters
fig, ax = plt.subplots()
ax.plot(range(len(accs_list)), accs_list, 'k', label='accuracy for BasicLSTMCell')
ax.legend(loc='upper right', shadow=True)
plt.savefig('./BasicLSTMCell_acc.png')
plt.show()

fig, bx = plt.subplots()
bx.plot(range(len(losses_list)), losses_list, 'k', label='loss for BasicLSTMCell')
bx.legend(loc='upper right', shadow=True)
plt.savefig('./BasicLSTMCell_loss.png')
plt.show()


