"""
Table Tennis position coaching assistant system using MPU-9150 dataset and LSTM RNNs by Se-Min Lim

This code is for test process

Bidirectional LSTM RNN
"""

import matplotlib
import itertools
import tensorflow as tf
import numpy as np
import time as t
import pruning_utils
import matplotlib.pyplot as plt
import math
import Bi_LSTM_Train as Bi
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
y_train = Bi.one_hot(Y_train)

# Read label test data
t_data_2 = np.loadtxt("DATA/Y_test.CSV", delimiter = ",", dtype = np.int32)
Y_test = t_data_2[:] - 1
y_test = Bi.one_hot(Y_test)

config = Bi.Config(X_train, X_test)
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("features shape, labels shape, each features mean, each features standard deviation")
print(X_test.shape, y_test.shape,
    np.mean(X_test), np.std(X_test))
print("the dataset is therefore properly normalised, as expected.")

X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_classes])

pred_Y = Bi.LSTM_Network(X, config)

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

pred_out, accuracy_out, loss_out = sess.run(
        [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )

# Predictions
predictions = pred_out.argmax(1)

# Confusion matrix figure
confusion_matrix = metrics.confusion_matrix(Y_test, predictions)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print(" Test accuracy : {}".format(accuracy_out))
print(" Lost : {}".format(loss_out))
print(" Precision: {}%".format(100*metrics.precision_score(Y_test, predictions, average="weighted")))
print(" Recall: {}%".format(100*metrics.recall_score(Y_test, predictions, average="weighted")))
print(" f1_score: {}%".format(100*metrics.f1_score(Y_test, predictions, average="weighted")))
print("")

# Print out confusion matrix
width = 12
height = 12
font = {
    'family' : 'Bitstream Vera Sans',
    'weight': 'bold',
    'size' : 15}
matplotlib.rc('font',**font)
plt.figure(figsize=(width, height))
plt.imshow(
    confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.Blues
)
tick_marks = np.arange(config.n_classes)
plt.xticks(tick_marks, LABELS, rotation = 90)
plt.yticks(tick_marks, LABELS)
fmt = 'd'
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True Label',size=25)
plt.xlabel('Predicted Label',size=25)
plt.show()

# Quantization
variable_names = [v.name for v in tf.trainable_variables()]

value_0 = sess.run(variable_names[0])
values_0, s_0 = Quant_Utils.quantize(value_0)
value_1 = sess.run(variable_names[1])
values_1, S_1 = Quant_Utils.quantize(value_1)
value_2 = sess.run(variable_names[2])
values_2, S_2 = Quant_Utils.quantize(value_2)
value_3 = sess.run(variable_names[3])
values_3, S_3 = Quant_Utils.quantize(value_3)
value_4 = sess.run(variable_names[4])
values_4, S_4 = Quant_Utils.quantize(value_4)
value_5 = sess.run(variable_names[5])
values_5, S_5 = Quant_Utils.quantize(value_5)
value_6 = sess.run(variable_names[6])
values_6, S_6 = Quant_Utils.quantize(value_6)
value_7 = sess.run(variable_names[7])
values_7, S_7 = Quant_Utils.quantize(value_7)


tf.dtypes.cast(value_0, tf.int32)
tf.dtypes.cast(value_1, tf.int32)
tf.dtypes.cast(value_2, tf.int32)
tf.dtypes.cast(value_3, tf.int32)
tf.dtypes.cast(value_4, tf.int32)
tf.dtypes.cast(value_5, tf.int32)
tf.dtypes.cast(value_6, tf.int32)
tf.dtypes.cast(value_7, tf.int32)


value_0 = values_0
value_1 = values_1
value_2 = values_2
value_3 = values_3
value_4 = values_4
value_5 = values_5
value_6 = values_6
value_7 = values_7

pred_out, accuracy_out, loss_out = sess.run(
[pred_Y, accuracy, cost],
    feed_dict={
        X: X_test,
        Y: y_test
    }
)

# Re-evaluate accuracy after weight quantization
print("")
print("final test accuracy: {}%".format(accuracy_out))
print("Precision: {}%".format(100*metrics.precision_score(Y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(Y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(Y_test, predictions, average="weighted")))
print("")
for v in value_1:
    print(v)

"""
dataframe = pd.DataFrame(value_1)
dataframe.to_csv("~/WORK/LSTM-RNN-Table-Tennis/Quant_Weight.csv", header = False, index = False)
"""

