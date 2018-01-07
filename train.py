import tensorflow as tf
from tensorflow import layers
import numpy as np
from database import SudokuMNIST

num_classes = 10
TERMINAL_PATH = './saved_model/sudoku_cv.ckpt'
normalize = True

# Prepare the Sudoku images for training
mnist = SudokuMNIST()
mnist.traintestsplit()

x = tf.placeholder(tf.float32, [None, 200, 200], name='X')
y = tf.placeholder(tf.int64, [None], name='Y')

isTrain = tf.placeholder(tf.bool, name='dropout_flag')

def deepnn(x, height, width, keep_prob=0.8):


    _x = tf.reshape(x, shape=[-1, height, width, 1])

    # CONV30 -> MAX_POOL2 ->
    out = layers.conv2d(_x, filters=30, kernel_size=5, activation=tf.nn.relu)
    out = layers.max_pooling2d(out, pool_size=2, strides=2)

    # -> CONV15 + MAX_POOL2 -> 
    out = layers.conv2d(out, filters=15, kernel_size=3, activation=tf.nn.relu)
    out = layers.max_pooling2d(out, pool_size=2, strides=2)
    
    # -> Dropout -> Flatten ->
    out = layers.dropout(out, rate=keep_prob, training=isTrain)
    out = layers.flatten(out)

    # -> Dense128 -> Dense50 -> Dense10
    out = layers.dense(out, units=128, activation=tf.nn.relu)
    out = layers.dense(out, units=50, activation=tf.nn.relu)
    return layers.dense(out, units=num_classes, name='logits_op')

logits = deepnn(x, 200, 200)

y_onehot = tf.one_hot(y, depth=num_classes)
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits)
entropy = tf.reduce_mean(entropy, name='entropy_op')

# Train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(entropy, name='minimize_op')

# Accuracy

softmax = tf.nn.softmax(logits, name='probability')

y_pred = tf.argmax(logits, 1, name='prediction_op')
prediction = tf.equal(y_pred, y)
prediction = tf.cast(prediction, tf.float32)
accuracy = tf.reduce_mean(prediction, name='accuracy_op')



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ACC = []
    steps = 0
    saver = tf.train.Saver(max_to_keep=None)

    while True:

        # Train using SGD
        x_train, y_train = mnist.next_batch(100)
        x_test, y_test = mnist.next_batch(50, False)

        if normalize:
            x_train = x_train / 255.
            x_test = x_test / 255.

        acc = sess.run(accuracy, feed_dict={
            x: x_train,
            y: y_train,
            isTrain: False
        })

        test_acc = sess.run(accuracy, feed_dict={
            x: x_test,
            y: y_test,
            isTrain: False
        })

        _, loss = sess.run([train_step, entropy],feed_dict={
            x: x_train, 
            y: y_train,
            isTrain: True
        } )
        if steps % 1 == 0:
            print ("Step:%s Accuracy %s(Train) %s(Test)"%(steps, acc, test_acc))
            print ("step %s Loss: %s"%(steps, loss))
        
        steps += 1
        ACC.append(acc)

        if np.mean(ACC[-50:]) >= 0.99 and test_acc >= 0.99:
            break
    
    # Save the model
    saver.save(sess, TERMINAL_PATH)

        