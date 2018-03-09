import numpy as np
import tensorflow as tf
import paint_auth as pa
from sklearn.model_selection import train_test_split, StratifiedKFold

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W, padding='SAME'):
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
    return tf.nn.avg_pool(x, ksize=ksize, strides=[1,1,1,1], padding=padding)

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
    
def model(x, num_class):
    #Conv. layer 1
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])
    
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    #Conv. layer 2
    W_conv2 = weight_variable([3,3,64,128])
    b_conv2 = bias_variable([128])
    
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #print h_conv2.shape
    #Conv. layer 3
    W_conv3 = weight_variable([3,3,128,256])
    b_conv3 = bias_variable([256])
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    
    #Conv. layer 4
    W_conv4 = weight_variable([3,3,256,256])
    b_conv4 = bias_variable([256])
    
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    
    #Conv. layer 5
    W_conv5 = weight_variable([3,3,256,128])
    b_conv5 = bias_variable([128])
    
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    
    b, w, l, d = h_conv5.shape
    
    #Avg. pooling
    #3x3 stride 1
    h_pool = avg_pool(h_conv5, [1,int(w),int(l),1], 'VALID')
    
    b, w, l, d = h_pool.shape #getting shape of conv5
    flat_dim = int(w*l*d)
    
    #fully connected
    W_fc1 = weight_variable([flat_dim, 2048])
    b_fc1 = bias_variable([2048])
    
    #reshape h_pool2
    h_pool2_flat = tf.reshape(h_pool, [-1, flat_dim])
    #fc 1
    #apply relu
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #fc 2
    W_fc2 = weight_variable([2048, 2048])
    b_fc2 = bias_variable([2048])    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
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
    
def run_model(X, num_classes):
    """
    Run given nerual network given input X and output y
    """
        
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
        output = model(x, num_classes)
        
        sess.run(tf.global_variables_initializer())
        sess.run(output, feed_dict={x: X})

def next_batch(num, data, labels):
    """
    Return a total of num random samples and labels. 
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
        
def train(X, y):
    """
    Trains model given data X and y labels
    """
    # Parameters
    learning_rate = 0.005
    training_epochs = 3
    batch_size = 10
    display_step = 1
    
    n = 256
    m = 256
    num_classes = 1179
    
    with tf.Session() as sess:
        
        #place holders for data
        x = tf.placeholder(tf.float32, shape=[None, n, m, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, num_classes])        

        #initialization
        output = model(x, num_classes)
        cost = loss(output, y_)
        train_step = train(cost, learning_rate)
        eval_op = evaluate(output, y_)
        
        sess.run(tf.global_variables_initializer())
        
        for i in range(training_epochs):
            batch = next_batch(batch_size, X, y)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], num_classes: num_classes})
            
            if i % display_step == 0:
                train_accuracy = sess.run(eval_op, feed_dict={
                    x:batch[0], y_: batch[1], num_classes: num_classes})
                print("step %d, training accuracy %g"%(i, train_accuracy))
        
        
    
    
if __name__ == "__main__":
    pass