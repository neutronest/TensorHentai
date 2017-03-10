import tensorflow as tf
import numpy as np

# download mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

OUTPUT_LAYER_NUM = 10


def cnn_flow(data):
    weights = {"w_conv1": tf.Variable(tf.random_normal([5, 5, 1, 32])),
               "w_conv2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
               "w_fc": tf.Variable(tf.random_normal([7*7*64, 1024])),
               "out": tf.Variable(tf.random_normal([1024, OUTPUT_LAYER_NUM]))}

    biases = {"b_conv1": tf.Variable(tf.random_normal([32])),
              "b_conv2": tf.Variable(tf.random_normal([64])),
              "b_fc": tf.Variable(tf.random_normal([1024])),
              "b_out": tf.Variable(tf.random_normal([OUTPUT_LAYER_NUM]))}
    data_reshape = tf.reshape(data, [-1, 28, 28, 1])
    h1 = tf.add(tf.nn.conv2d(data_reshape, weights["w_conv1"], strides=[1,1,1,1], padding="SAME"), biases["b_conv1"])
    conv1 = tf.nn.relu(h1)
    conv1_with_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    h2 = tf.add(tf.nn.conv2d(conv1_with_pool, weights["w_conv2"], strides=[1,1,1,1], padding="SAME"), biases["b_conv2"])
    conv2 = tf.nn.relu(h2)
    conv2_with_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    conv2_with_pool_reshape = tf.reshape(conv2_with_pool, [-1, 7*7*64])
    h3 = tf.add(tf.matmul(conv2_with_pool_reshape, weights["w_fc"]), biases["b_fc"])
    fc = tf.nn.relu(h3)
    # YOU CAN USE DROPOUT here
    output = tf.add(tf.matmul(fc, weights["out"]), biases["b_out"])
    return output

def main():
    mnist_data = input_data.read_data_sets("/tmp/", one_hot=True)
    X = tf.placeholder("float", [None, 28*28])
    Y = tf.placeholder("float")
    batch_size = 100

    y_pred = cnn_flow(X)
    cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_fn)
    epoches = 20
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in xrange(0, epoches):
            epoch_loss = 0
            for i in xrange(0, int(mnist_data.train.num_examples/batch_size)):
                x, y = mnist_data.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost_fn], feed_dict={X:x, Y:y})
                epoch_loss += c
            print "epoch ", epoch, ":", (epoch_loss * 1.0 / 100)
            correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            print "accuracy: ", accuracy.eval({X:mnist_data.test.images, Y:mnist_data.test.labels})
        
    return



if __name__ == "__main__":
    main()
