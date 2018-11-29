import tensorflow as tf
mnist = tf.keras.datasets.mnist
nb_classes = 10

train, test = mnist.load_data()

(x_train, y_train) = train
(x_train, y_train) = (x_train/255.0, y_train)
x_train = tf.reshape(x_train, [-1, 28*28])
y_train = tf.one_hot(y_train, nb_classes)

(x_test, y_test) = test
(x_test, y_test) = (x_test/255.0, y_test)
x_test = tf.reshape(x_test, [-1, 28*28])
y_test = tf.one_hot(y_test, nb_classes)

data_rows, data_cols = x_train.shape


batch_size = 100
training_epochs = 15

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(100000).repeat().batch(batch_size=batch_size)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

test_data = tf.data.Dataset.from_tensors((x_test, y_test))
test_iter = test_data.make_one_shot_iterator()


''' Model '''

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([28*28, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits=logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_batch_test, y_batch_test = sess.run(test_iter.get_next())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(int(data_rows) / batch_size)

        for i in range(total_batch):
            x_batch, y_batch = sess.run(next_element)
            c, _ = sess.run([cost, optimizer], {X: x_batch, Y: y_batch})
            avg_cost += c/total_batch
        print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:x_batch_test, Y:y_batch_test}))