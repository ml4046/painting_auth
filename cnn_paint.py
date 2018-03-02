import tensorflow as tf


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W, padding='VALID'):
    """
    strides = [b,i,j,d]
    b: starting batch
    d: depth of batch
    here we have from 1 to 1 depth so every image is used
    """
    
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def avg_pool(x, ksize, padding):
    return tf.nn.avg_pool(x, ksize=ksize, strides=[1,2,2,1], padding=padding)

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def loss(output, y):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    return cross_entropy

def train(cost, l_rate):
    train_op = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)
    return train_op
    
def model(x, num_classes):
    #Conv. layer 1
    W_conv1 = weight_variable([3, 3, 1, 12])
    b_conv1 = bias_variable([12])
    
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    #Conv. layer 2
    W_conv2 = weight_variable([3,3,12,64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, padding='SAME') + b_conv2)
    
    #Conv. layer 3
    W_conv3 = weight_variable([3,3,64,128])
    b_conv3 = bias_variable([128])
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    
    #Conv. layer 4
    W_conv4 = weight_variable([3,3,128,256])
    b_conv4 = bias_variable([256])
    
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    
    #Conv. layer 5
    W_conv5 = weight_variable([3,3,256,256])
    b_conv5 = bias_variable([256])
    
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    
    #Conv. layer 6
    W_conv6 = weight_variable([3,3,256,128])
    b_conv6 = bias_variable([128])
    
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)     
    
    #Avg. pooling
    #3x3 stride 1
    h_pool = avg_pool(h_conv6, [1,3,3,1], 'VALID')
    
    #fully connected
    W_fc1 = weight_variable([5*5*128, 2048])
    b_fc1 = bias_variable([2048])
    
    #reshape h_pool2
    h_pool2_flat = tf.reshape(h_pool, [-1, 5*5*128])
    #fc 1
    #apply relu
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #fc 2
    W_fc2 = weight_variable([2048, 2048])
    b_fc2 = bias_variable([2048])    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc1) + b_fc1)
    
    #output layer
    W_out = weight_variable([2048, num_class])
    b_out = bias_variable([num_class])
    output = tf.nn.relu(tf.matmul(h_fc2, W_out) + b_out)
    
    return output
    
    ##applying dropout
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #W_fc2 = weight_variable([1024, 10])
    #b_fc2 = bias_variable([10])
    
    #y_hat = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
if __name__ == "__main__":
    
    # Parameters
    learning_rate = 0.005
    
    training_epochs = 2000
    batch_size = 100
    
    display_step = 10
    
    with tf.Session() as sess:

        #place holders for data
        x = tf.placeholder(tf.float32, shape=[None, 784])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])        

        #initialization
        keep_prob = tf.placeholder(tf.float32)
        output = model(x_image, keep_prob)
        cost = loss(output, y_)
        train_step = train(cost, learning_rate)
        eval_op = evaluate(output, y_)
        
        sess.run(tf.global_variables_initializer())
        
        for i in range(training_epochs):
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            
            if i % display_step == 0:
                train_accuracy = sess.run(eval_op, feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
        
        print('test accuracy %g' % sess.run(eval_op, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))