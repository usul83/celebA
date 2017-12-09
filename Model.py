import tensorflow as tf
import os

def weight_variable(name, shape, sd=0.0001):
    initial = tf.truncated_normal_initializer(stddev=sd)
    return tf.get_variable(name,shape,initializer=initial)


def bias_variable(name, shape, value=0.0001):
    initial = tf.constant_initializer(value)
    return tf.get_variable(name,shape,initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


class Model:
    def __init__(self, imagesIn, labelsTru, convWidth, conv1feat, conv2feat, conv3feat, linear1feat):
        # get dimensions
        heightIn = int(imagesIn.get_shape()[1])
        widthIn = int(imagesIn.get_shape()[2])

        self.images = imagesIn
        self.labelsTru = labelsTru
        self.linear1feat = linear1feat

        # define graph variables
        Wconv1 = weight_variable('Wconv1',[convWidth, convWidth, 3, conv1feat], 1e-4)
        bConv1 = bias_variable('bconv1',[conv1feat])
        Wconv2 = weight_variable('Wconv2',[convWidth, convWidth, conv1feat, conv2feat], 0.01)
        bConv2 = bias_variable('bconv2',[conv2feat])
        Wconv3 = weight_variable('Wconv3',[convWidth, convWidth, conv2feat, conv3feat], 0.01)
        bConv3 = bias_variable('bconv3',[conv3feat])
        width3 = int(widthIn / 2 / 2 /2)
        height3 = int(heightIn / 2 / 2 / 2)
        Wfc1 = weight_variable('Wfc1',[width3 * height3 * conv3feat, linear1feat], 0.1)
        bFC1 = bias_variable('bfc1',[linear1feat])

        #downsample image
        xPool0 = max_pool(imagesIn)

        # convolution 1, with batch-norm and pooling
        hConv1pre = conv2d(xPool0, Wconv1) + bConv1
        hConv1n = tf.contrib.layers.batch_norm(hConv1pre, center=True, scale=True)
        hConv1n = tf.nn.relu(hConv1n)
        # hConv1n = tf.nn.dropout(hConv1n,1-hiddenDropRate)
        hPool1 = max_pool(hConv1n)
        # hPool1 = hConv1

        # convolution 2, with batch-norm and no pooling
        hConv2pre = conv2d(hPool1, Wconv2) + bConv2
        hConv2n = tf.contrib.layers.batch_norm(hConv2pre, center=True, scale=True)
        hConv2n = tf.nn.relu(hConv2n)
        # hConv2n = tf.nn.dropout(hConv2n, 1-hiddenDropRate)
        # hPool2 = max_pool(hConv2n)
        hPool2 = hConv2n

        # convolution3 with skip connection to output from conv1, followed by pooling
        hConv3pre = conv2d(hPool2, Wconv3) + bConv3 + hPool1
        hConv3n = tf.contrib.layers.batch_norm(hConv3pre, center=True, scale=True)
        hConv3n = tf.nn.relu(hConv3n)
        hPool3 = max_pool(hConv3n)

        # non-linear layer 1, flatten processed image into vector; transform and apply batch norm and relu
        hPool2flat = tf.reshape(hPool3, [-1, width3 * height3 * conv3feat])
        hFC1pre = tf.matmul(hPool2flat, Wfc1) + bFC1
        hFC1n = tf.contrib.layers.batch_norm(hFC1pre, center=True, scale=True)
        self.hFC1n = tf.nn.relu(hFC1n)

        self.saver = tf.train.Saver()

    def _detect_attributes(self,numLabels):
        # training options for detecting face attributes
        classes = numLabels - 4
        self.Wfc2 = weight_variable('Wfc2',[self.linear1feat, classes], 0.1)
        self.bFC2 = bias_variable('bfc2',[classes])

        # readout layer, softmax means elements of y form probability distribution (sum to one)
        self.labels = tf.nn.sigmoid(tf.matmul(self.hFC1n, self.Wfc2) + self.bFC2)
        self.labelsOH = tf.stack([self.labels, 1-self.labels],axis=-1)

        # calculate cross entropy between predictions and ground truth
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(self.labelsTru[:,4:],2) *
                                                          tf.log(self.labelsOH), reduction_indices=[1,2]))
        correct_prediction = tf.equal(tf.round(self.labels), tf.cast(self.labelsTru[:,4:],tf.float32))

        # correct_prediction gives list of booleans, take mean to measure % accuracy
        self.accuracies = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),axis=0)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _detect_bb(self,cropWidth,cropHeight,batchSize):
        # training options for learning to generate boundary boxes
        classes = 4
        self.Wfc2 = weight_variable('Wfc2',[self.linear1feat, classes], 0.1)
        self.bFC2 = bias_variable('bfc2',[classes])

        # readout layer, softmax means elements of y form probability distribution (sum to one)
        self.labels = tf.abs(tf.matmul(self.hFC1n, self.Wfc2) + self.bFC2)
        # self.labelsWidth = tf.stack([self.labels[:,2]-self.labels[:,0], self.labels[:,3]-self.labels[:,1]],axis=-1)
        scaling = tf.constant([1/cropWidth,1/cropHeight,1/cropWidth,1/cropHeight])

        #ammend labels to give end poisitons as well as width
        self.labelsBBTru = self.labelsTru[:,:4] + tf.concat([tf.constant(0, dtype=tf.int64, shape=[batchSize, 2]),
                                                    self.labelsTru[:, :2]], axis=1)
        self.labelsBB = self.labels[:, :4] + tf.concat([tf.constant(0, dtype=tf.float32, shape=[batchSize, 2]),
                                                           self.labels[:, :2]], axis=1)
        # self.labelsBB = tf.concat([self.labelsBB,self.labelsTru[:,2:4]], axis=1)

        # calculate mean squared error between predictions and ground truth
        self.loss = tf.losses.mean_squared_error(self.labels, tf.cast(self.labelsTru[:,:4],tf.float32) * scaling) +\
                    tf.losses.mean_squared_error(self.labelsBB[:,2:4], tf.cast(self.labelsBBTru[:,2:4],tf.float32) /cropWidth)

        # draw bounding boxes onto batch (change to tf syntax first)
        self.BBtfFormat = self.labels + tf.concat([tf.constant(0, dtype=tf.float32, shape=[batchSize, 2]),
                                                   self.labels[:, :2]], axis=1)
        self.BBtfFormatS = tf.cast(self.BBtfFormat[:,None,:4],tf.float32)
        self.BBtfFormatS = tf.stack([self.BBtfFormatS[:,:,1],self.BBtfFormatS[:,:,0],self.BBtfFormatS[:,:,3],
                                     self.BBtfFormatS[:,:,2]],axis=-1)
        self.BBimages = tf.image.draw_bounding_boxes(self.images, self.BBtfFormatS)

    def _use_nesterov(self, eta=0.1, gamma=0.9):
        # set training method
        self.trainStep = tf.train.MomentumOptimizer(eta, gamma, use_nesterov=True).minimize(self.loss)

    def _add_tensorboard(self, logPath, sess):
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(logPath + '/train', sess.graph)
        self.test_writer = tf.summary.FileWriter(logPath + '/test', sess.graph)

    def restore(self,session,checkpoint):
        self.saver.restore(session, checkpoint)

