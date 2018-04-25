import numpy as np
import tensorflow as tf

def weight_variables(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variables(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W, kernel_size, strides, padding='SAME'):
    """
    strides = [b,i,j,d]
    b: starting batch
    d: depth of batch
    """
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

def batch_norm(x, training):
    return tf.layers.batch_normalization(
         inputs=x, axis=3,
         momentum=.99, epsilon=1e-5, center=True,
         scale=True, training=training, fused=True)    

def build_block(x, num_filters, kernel_size, stride_size, training, downsize=False):
    """
    Creates a block of layers for Resnet
    """
    in_channel = x.get_shape().as_list()[-1]
    strides = [1,stride_size, stride_size,1]
    out_channel = num_filters
    
    #shortcut
    #testing in_channel vs out_channel size
    if in_channel == out_channel:
        if downsize:
            shortcut = tf.nn.max_pool(x, [1,stride_size, stride_size,1], [1,stride_size, stride_size,1],'VALID')
        else:
            shortcut = x
    else:
        Ws = weight_variables([1,1,in_channel, out_channel])
        #downsampling by half (strides 2), out_channel doubles
        shortcut = conv2d(x, Ws, 1, [1,2,2,1])
        shortcut = batch_norm(shortcut, training)
    
    #layer 1
    W1 = weight_variables([kernel_size, kernel_size, in_channel, num_filters])
    b1 = bias_variables([num_filters])
    h1 = batch_norm(conv2d(x, W1, kernel_size, strides) + b1, training)
    h1 = tf.nn.relu(h1)
    #print 'h1 shape: '
    #print h1.shape
    
    #layer 2 No downsample on 2nd block, strides=[1,1,1,1]
    W2 = weight_variables([kernel_size, kernel_size, num_filters, num_filters])
    b2 = bias_variables([num_filters])
    h2 = batch_norm(conv2d(h1, W2, kernel_size, [1,1,1,1]) + b2, training)
    ##insert shortcut x
    #print 'shortcut shape: '
    #print shortcut.shape
    #print 'h2 shape: '
    #print h2.shape
    
    y = tf.nn.relu(h2 + shortcut)
    return y

class Resnet(object):
    def __init__(self, img_shape, n_classes, optimizer, sess, reuse_weights=False):
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.sess = sess
        self.output = None
        ##initialize boolean tensor to specify training or not
        self.training = tf.placeholder(tf.bool)
        h, w, d = self.img_shape
        self.images = tf.placeholder(tf.float32, [None, h, w, d])        
        
    def build_model(self, n_filters=[64, 128, 256, 512], s_kernels=[3, 3, 3, 3], strides=[2, 2, 2, 2]):
        """
        params:
            n_filters: array of size of filters in each block
            s_kernels: array of size of kernels in each conv. layer in res. block
            strides: array of size of strides in each conv. layer in res. block
        """
        n_filters = [64, 128, 256, 512]
        kernels = [3, 3, 3, 3]
        strides = [2, 2, 2, 2]
        
        #conv and maxpool layer
        W0 = weight_variables([7, 7, 3, 64])
        b0 = bias_variables([64])
        h0 = max_pool_3x3(tf.nn.relu(batch_norm(conv2d(self.images, W0, 7, [1,2,2,1]) + b0, self.training)))
        
        #res. blocks
        ##block 1
        b1 = build_block(h0, n_filters[0], s_kernels[0], 1, self.training)
        b2 = build_block(b1, n_filters[0], s_kernels[0], 1, self.training)
        ##blocks 2
        b3 = build_block(b2, n_filters[1], s_kernels[1], strides[1], self.training)
        b4 = build_block(b3, n_filters[1], s_kernels[1], 1, self.training)
        ##block 3
        b5 = build_block(b4, n_filters[2], s_kernels[2], strides[2], self.training)
        b6 = build_block(b5, n_filters[2], s_kernels[2], 1, self.training)
        ##block4
        b7 = build_block(b6, n_filters[3], s_kernels[3], strides[3], self.training)
        b8 = build_block(b7, n_filters[3], s_kernels[3], 1, self.training)
        
        #fc layer
        ##avg pooling can be done with with reduce_mean
        pool = tf.reduce_mean(b8, [1,2], keepdims=True)
        pool = tf.reshape(pool, [-1,pool.shape[-1]])
        pool = tf.layers.dense(pool, units=1000, activation=tf.nn.softmax)
        output = tf.layers.dense(pool, units=self.n_classes, activation=tf.nn.softmax)
        self.output = output
        
    def build_loss(self):
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        print self.output.shape
        xentropy = tf.losses.softmax_cross_entropy(self.labels, self.output)
        self.train_op = self.optimizer.minimize(xentropy)
        
    def train(self, images, labels):
        return self.sess.run(self.train_op, feed_dict={self.images:images, self.labels:labels, self.training:True})
    
    def predict(self, images):
        return self.sess.run(self.output, feed_dict={self.images:images, self.training:False})
        
        
    
        
        