import tensorflow as tf
import input_data

# Get mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Get the interactive session
sess = tf.InteractiveSession()

# Input image data
# shape:None -> dynamic allocate 
# shape:784 -> 28x28
x = tf.placeholder("float", shape=[None, 784])

# Output
# shape:None -> dynamic allocate
# shape:10 -> 10 dimention
y_ = tf.placeholder("float", shape=[None, 10])


# Weight formatting
def weight_variable(shape):
    # Generate random number from normal distribution
    # Calcurate var
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Bias formatting
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Generate convolution
# x:input, shape [batch, in_height, in_width, in_channels]
# W:filter, shape [filter_height, filter_width, in_channels, out_channels]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Generate max_pooling
# x:input, shape [batch, in_height, in_width, in_channels]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Initial value of weight and bias
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Image
x_image = tf.reshape(x, [-1,28,28,1])

# Create some layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Training data
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# softmax( h_fc1_drop * W_fc2 + b_fc2 )
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Loss -> cross entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# BP
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Calcurate accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Start session and initialize
sess.run(tf.initialize_all_variables())
print "start training"
# Start training
for i in range(200):
    # Select 50 data 
    batch = mnist.train.next_batch(50)
    # Calcurate train accuracy
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images[0:1000, ], y_: mnist.test.labels[0:1000, ], keep_prob: 1.0}))
