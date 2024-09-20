import scipy.io
import numpy as np
import tensorflow as tf

# Load the .mat file
mat_data = scipy.io.loadmat('file.mat')

# Assuming 'X' is the input data and 'y' is the labels in the .mat file
X = mat_data['X']
y = mat_data['y']

# Normalize the input data
X = X.astype('float32') / 255.0

# Convert labels to categorical
y = tf.one_hot(y.flatten(), depth=np.max(y) + 1)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model using TensorFlow
model = tf.Graph()
with model.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None] + list(X_train.shape[1:]), name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, y_train.shape[1]], name='labels')

    conv1 = tf.layers.conv2d(inputs, 32, (3, 3), activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2))

    conv2 = tf.layers.conv2d(pool1, 64, (3, 3), activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2))

    flat = tf.layers.flatten(pool2)
    dense = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    logits = tf.layers.dense(dense, y_train.shape[1])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train the model
with tf.Session(graph=model) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        _, train_loss, train_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={inputs: X_train, labels: y_train})
        print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy}')

    # Evaluate the model
    test_accuracy = sess.run(accuracy, feed_dict={inputs: X_test, labels: y_test})
    print(f'Test accuracy: {test_accuracy}')