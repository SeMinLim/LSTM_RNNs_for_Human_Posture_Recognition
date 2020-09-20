"""
Table Tennis position coaching assistant system using MPU-9150 dataset and LSTM RNNs by Se-Min Lim

This code is for pruning process

Unidirectional LSTM RNN
"""
import pandas as pd
import matplotlib
import itertools
import tensorflow as tf
import numpy as np
import time as t
import pruning_utils
import Quant_Utils
import matplotlib.pyplot as plt
import math
import Uni_LSTM_Train as Uni
from sklearn import metrics

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

# Read train data file
x_train= np.loadtxt("DATA/X_train.CSV", delimiter = ",", dtype = np.float32) # 1260x27 matrics
X_train_t = np.vstack([[x_train[:70,:]],[x_train[70:140,:]],[x_train[140:210,:]],[x_train[210:280,:]],[x_train[280:350,:]],[x_train[350:420,:]],
                    [x_train[420:490,:]],[x_train[490:560,:]],[x_train[560:630,:]],[x_train[630:700,:]],[x_train[700:770,:]],[x_train[770:840,:]],
                    [x_train[840:910,:]],[x_train[910:980,:]],[x_train[980:1050,:]],[x_train[1050:1120,:]],[x_train[1120:1190,:]],[x_train[1190:1260,:]]])
X_train = np.transpose(X_train_t, (1,2,0))

# Read test data file
x_test = np.loadtxt("DATA/X_test.CSV", delimiter = ",", dtype = np.float32) # 540x27 matrics
X_test_t = np.vstack([[x_test[:30,:]],[x_test[30:60,:]],[x_test[60:90,:]],[x_test[90:120,:]],[x_test[120:150,:]],[x_test[150:180,:]],[x_test[180:210,:]],
                    [x_test[210:240,:]],[x_test[240:270,:]],[x_test[270:300,:]],[x_test[300:330,:]],[x_test[330:360,:]],[x_test[360:390,:]],[x_test[390:420,:]],
                    [x_test[420:450,:]],[x_test[450:480,:]],[x_test[480:510,:]],[x_test[510:540,:]]])
X_test = np.transpose(X_test_t,(1,2,0))

# Read label train data
t_data_1 = np.loadtxt("DATA/Y_train.CSV", delimiter = ",", dtype = np.int32)
Y_train = t_data_1[:] - 1
y_train = Uni.one_hot(Y_train)

# Read label test data
t_data_2 = np.loadtxt("DATA/Y_test.CSV", delimiter = ",", dtype = np.int32)
Y_test = t_data_2[:] - 1
y_test = Uni.one_hot(Y_test)

# Fine-tuning 
config = Uni.Config(X_train, X_test)
config.learning_rate = 0.00025
config.lambda_loss_amount = 0.0001
config.training_epochs = 50
config.batch_size = 5

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("features shape, labels shape, each features mean, each features standard deviation")
print(X_test.shape, y_test.shape,
    np.mean(X_test), np.std(X_test))
print("the dataset is therefore properly normalised, as expected.")

X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_classes])

pred_Y = Uni.LSTM_Network(X, config)

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

sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU':0}))
saver.restore(sess, config.save_file)

# Pruning
variable_names = [v.name for v in tf.trainable_variables()]

value_0 = sess.run(variable_names[0])
values_0 = pruning_utils.prune_weights(value_0, 1.0)
value_1 = sess.run(variable_names[1])
values_1 = pruning_utils.prune_weights(value_1, 1.0)
value_2 = sess.run(variable_names[2])
values_2 = pruning_utils.prune_weights(value_2, 1.0)
value_3 = sess.run(variable_names[3])
values_3 = pruning_utils.prune_weights(value_3, 1.0)
value_4 = sess.run(variable_names[4])
values_4 = pruning_utils.prune_weights(value_4, 1.0)
value_5 = sess.run(variable_names[5])
values_5 = pruning_utils.prune_weights(value_5, 1.0)
value_6 = sess.run(variable_names[6])
values_6 = pruning_utils.prune_weights(value_6, 1.0)
value_7 = sess.run(variable_names[7])
values_7 = pruning_utils.prune_weights(value_7, 1.0)
graph = tf.get_default_graph()
w0 = graph.get_tensor_by_name("Variable:0")
w1 = graph.get_tensor_by_name("Variable_1:0")
w2 = graph.get_tensor_by_name("Variable_2:0")
w3 = graph.get_tensor_by_name("Variable_3:0")
w4 = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0")
w5 = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0")
w6 = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0")
w7 = graph.get_tensor_by_name("rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0")
feed_dict = {w0:values_0, w1:values_1, w2:values_2, w3:values_3,
            w4:values_4, w5:values_5, w6:values_6, w7:values_7}

# Except zero-weights on re-training session
pruning_utils.mask_for_big_values(value_0, 1.0)
pruning_utils.mask_for_big_values(value_1, 1.0)
pruning_utils.mask_for_big_values(value_2, 1.0)
pruning_utils.mask_for_big_values(value_3, 1.0)
pruning_utils.mask_for_big_values(value_4, 1.0)
pruning_utils.mask_for_big_values(value_5, 1.0)
pruning_utils.mask_for_big_values(value_6, 1.0)
pruning_utils.mask_for_big_values(value_7, 1.0)

best_accuracy = 0.0

# Start re-training for each batch and loop epochs
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
    end_time = t.time() - start_time
    print("traing iter: {},".format(i) +
          " test accuracy : {},".format(accuracy_out) +
          " loss : {},".format(loss_out) + " use time : {}".format(end_time))
    best_accuracy = max(best_accuracy, accuracy_out)

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

# Print out weight_0
for v in value_0:
    print(v)

# Quantization

q_0, s_0 = Quant_Utils.quantize(value_0)
q_1, S_1 = Quant_Utils.quantize(value_1)
q_2, S_2 = Quant_Utils.quantize(value_2)
q_3, S_3 = Quant_Utils.quantize(value_3)
q_4, S_4 = Quant_Utils.quantize(value_4)
q_5, S_5 = Quant_Utils.quantize(value_5)
q_6, S_6 = Quant_Utils.quantize(value_6)
q_7, S_7 = Quant_Utils.quantize(value_7)


tf.dtypes.cast(value_0, tf.int8)
tf.dtypes.cast(value_1, tf.int8)
tf.dtypes.cast(value_2, tf.int8)
tf.dtypes.cast(value_3, tf.int8)
tf.dtypes.cast(value_4, tf.int8)
tf.dtypes.cast(value_5, tf.int8)
tf.dtypes.cast(value_6, tf.int8)
tf.dtypes.cast(value_7, tf.int8)


value_0 = q_0
value_1 = q_1
value_2 = q_2
value_3 = q_3
value_4 = q_4
value_5 = q_5
value_6 = q_6
value_7 = q_7

# Print out weight_0
for v in value_0:
    print(v)

# Re-evaluate accuracy after weight quantization
pred_out, accuracy_out, loss_out = sess.run(
    [pred_Y, accuracy, cost],
    feed_dict={
        X: X_test,
        Y: y_test
    }
)
print("")
print("final test accuracy: {}%".format(accuracy_out))
print("Precision: {}%".format(100*metrics.precision_score(Y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(Y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(Y_test, predictions, average="weighted")))
print("")

"""
dataframe = pd.DataFrame(value_1)
dataframe.to_csv("~/WORK/LSTM-RNN-Table-Tennis/Quant_Weight.csv", header = False, index = False)
"""
