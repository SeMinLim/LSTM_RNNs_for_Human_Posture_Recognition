"""
Table Tennis Posture Coaching Assistant System using MPU-9150 and LSTM RNNs by Se-Min Lim

This code is for training process

Unidirectional LSTM RNN
"""

import matplotlib
import itertools
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import math
from sklearn import metrics

class Config(object):
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 1260(7x18x5x2) training series
        self.test_data_count = len(X_test)  # 540(3x18x5x2) testing series
        self.n_steps = len(X_train[0])  # 27 time_steps per series

        # Training
        self.learning_rate = 0.00025
        self.lambda_loss_amount = 0.0001
        self.training_epochs = 200
        self.batch_size = 5
        self.save_file = './train_model_uni.ckpt'
        
        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 32  # nb of hidden layer inside the neural network
        self.n_classes = 10  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes])) # Bidirectional
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }

def LSTM_Network(_X, config):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps,0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']

def one_hot(y_):
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

    # Output classes to learn how to classify
    LABELS = [
    "Coach's Forehand Stroke",
	"Coach's Forehand Drive",
	"Coach's Forehand Cut",
	"Coach's Backhand Drive",
	"Coach's Backhand Short",
	"Beginner's Forehand Stroke",
	"Beginner's Forehand Drive",
	"Beginner's Forehand Cut",
	"Beginner's Backhand Drive",
	"Beginner's Backhand Short"
    ]

    x_train= np.loadtxt("DATA/X_train.CSV", delimiter = ",", dtype = np.float32) # 1260x27 matrics
    X_train_t = np.vstack([[x_train[:70,:]],[x_train[70:140,:]],[x_train[140:210,:]],[x_train[210:280,:]],[x_train[280:350,:]],[x_train[350:420,:]],
			[x_train[420:490,:]],[x_train[490:560,:]],[x_train[560:630,:]],[x_train[630:700,:]],[x_train[700:770,:]],[x_train[770:840,:]],
			[x_train[840:910,:]],[x_train[910:980,:]],[x_train[980:1050,:]],[x_train[1050:1120,:]],[x_train[1120:1190,:]],[x_train[1190:1260,:]]])
    X_train = np.transpose(X_train_t, (1,2,0))

    x_test = np.loadtxt("DATA/X_test.CSV", delimiter = ",", dtype = np.float32) # 540x27 matrics
    X_test_t = np.vstack([[x_test[:30,:]],[x_test[30:60,:]],[x_test[60:90,:]],[x_test[90:120,:]],[x_test[120:150,:]],[x_test[150:180,:]],[x_test[180:210,:]],
			[x_test[210:240,:]],[x_test[240:270,:]],[x_test[270:300,:]],[x_test[300:330,:]],[x_test[330:360,:]],[x_test[360:390,:]],[x_test[390:420,:]],
			[x_test[420:450,:]],[x_test[450:480,:]],[x_test[480:510,:]],[x_test[510:540,:]]])
    X_test = np.transpose(X_test_t,(1,2,0))

    t_data_1 = np.loadtxt("DATA/Y_train.CSV", delimiter = ",", dtype = np.int32)
    Y_train = t_data_1[:] - 1
    y_train = one_hot(Y_train)

    t_data_2 = np.loadtxt("DATA/Y_test.CSV", delimiter = ",", dtype = np.int32)
    Y_test = t_data_2[:] - 1
    y_test = one_hot(Y_test)
    
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    saver = tf.train.Saver()

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU':0}))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    test_accuracy = []
    test_cost = []

    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        start_time = t.time()
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
        [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
        test_accuracy.append(accuracy_out*100)
        test_cost.append(loss_out)
        end_time = t.time() - start_time
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {},".format(loss_out) + " use time : {}".format(end_time))
        best_accuracy = max(best_accuracy, accuracy_out)

    # Predictions
    predictions = pred_out.argmax(1)

    print("")
    print("final test accuracy: {}%".format(accuracy_out))
    print("best epoch's test accuracy: {}%".format(best_accuracy))
    print("")
 
    print("")
    print("Precision: {}%".format(100*metrics.precision_score(Y_test, predictions, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(Y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(Y_test, predictions, average="weighted")))
    print("")

    saver.save(sess, config.save_file)
    print("Save Success!")
   
    # Print out plot containing accuracy info
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,100])
    ax.set_xlim([0,200])
    ax.set_xticks([0,50,100,150,200])
    ax.plot(test_accuracy, color = 'blue', linewidth=3)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Training Iteration")
    plt.show()
