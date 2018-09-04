# tensorflow keras
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape, Conv2DTranspose
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate, MaxPooling2D
from tensorflow.contrib.keras.api.keras.backend import resize_images
from tensorflow.python.keras.utils import multi_gpu_model

class DispNet(object):
    def __init__(self, img_height, img_width, img_depth,  learning_rate, batch_size = 1, mode = 'correlation_'):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mode = mode

        self.model_in_height = self.resize(img_height)
        self.model_in_width = self.resize(img_width)
        self.model_in_depth = 3

        self.model_out_height = int(self.model_in_height / 2)
        self.model_out_width = int(self.model_in_width / 2)
        self.model_out_depth = 1
        
        self.left_image = tf.placeholder(tf.float32, [self.batch_size, self.model_in_height, self.model_in_width, self.model_in_depth], name = 'left_image')
        self.right_image = tf.placeholder(tf.float32, [self.batch_size, self.model_in_height, self.model_in_width, self.model_in_depth], name = 'right_image')
        self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.model_out_height, self.model_out_width, self.model_out_depth], name = 'ground_truth')

        print('input image resized by (height = %s,' %self.model_in_height, 'width = %s)' %self.model_in_width)

    def resize(self, value, multiple = 64):
        n = 1
        condition = False
        diff_before = value
        diff_after = 0

        while not condition:
            diff_after = abs(multiple * n - value)

            if diff_after < diff_before:
                diff_before = diff_after
            else:
                condition = True
                return multiple * (n - 1)
            n += 1
            
    def weight_variable(self, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.Variable(initializer(shape=shape), name='weight')

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='b')
            
    def DispNetCorr_(self, input_left, input_right, ground_truth):

        left_Conv1 = None
        with tf.name_scope('left_Conv1'):
            W = self.weight_variable([7, 7, 3, 64]) 
            b = self.bias_variable([64])
            left_Conv1 = tf.nn.relu(tf.nn.conv2d(input_left, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        left_Conv2 = None     
        with tf.name_scope('left_Conv2'):
            W = self.weight_variable([5, 5, 64, 128]) 
            b = self.bias_variable([128])
            left_Conv2 = tf.nn.relu(tf.nn.conv2d(left_Conv1, W, strides=[1,2,2,1], padding='SAME') + b)
                  
        right_Conv1 = None
        with tf.name_scope('right_Conv1'):
            W = self.weight_variable([7, 7, 3, 64]) 
            b = self.bias_variable([64])
            right_Conv1 = tf.nn.relu(tf.nn.conv2d(input_right, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        right_Conv2 = None
        with tf.name_scope('right_Conv2'):
            W = self.weight_variable([5, 5, 64, 128]) 
            b = self.bias_variable([128])
            right_Conv2 = tf.nn.relu(tf.nn.conv2d(right_Conv1, W, strides=[1,2,2,1], padding='SAME') + b)

        Corr_dispRange = None
        with tf.name_scope('Corr_dispRange'):
            x = left_Conv2
            y = right_Conv2
            max_disp = 40
            corr_tensors = []
            for i in range(-max_disp, 0, 1):
                shifted = tf.pad(tf.slice(y, [0]*4, [-1, -1, y.shape[2].value + i, -1]),
                                 [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
                corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
                corr_tensors.append(corr)
            for i in range(max_disp + 1):
                shifted = tf.pad(tf.slice(x, [0, 0, i, 0], [-1]*4),
                                 [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
                corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
                corr_tensors.append(corr)
            Corr_dispRange = tf.transpose(tf.stack(corr_tensors), perm=[1, 2, 3, 0])
                     
        Conv_redir = None
        with tf.name_scope('Conv_redir'):
            W = self.weight_variable([1, 1, 128, 64]) 
            b = self.bias_variable([64])
            Conv_redir = tf.nn.relu(tf.nn.conv2d(left_Conv2, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        Corr = tf.concat([Corr_dispRange, Conv_redir], axis=3, name='Corr')

        Conv3 = None
        with tf.name_scope('Conv3'):
            W = self.weight_variable([5, 5, 145, 256]) 
            b = self.bias_variable([256])
            Conv3 = tf.nn.relu(tf.nn.conv2d(Corr, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        Conv3_1 = None
        with tf.name_scope('Conv3_1'):
            W = self.weight_variable([3, 3, 256, 256]) 
            b = self.bias_variable([256])
            Conv3_1 = tf.nn.relu(tf.nn.conv2d(Conv3, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        Conv4 = None
        with tf.name_scope('Conv4'):
            W = self.weight_variable([3, 3, 256, 512]) 
            b = self.bias_variable([512])
            Conv4 = tf.nn.relu(tf.nn.conv2d(Conv3_1, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        Conv4_1 = None
        with tf.name_scope('Conv4_1'):
            W = self.weight_variable([3, 3, 512, 512]) 
            b = self.bias_variable([512])
            Conv4_1 = tf.nn.relu(tf.nn.conv2d(Conv4, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        Conv5 = None
        with tf.name_scope('Conv5'):
            W = self.weight_variable([3, 3, 512, 512]) 
            b = self.bias_variable([512])
            Conv5 = tf.nn.relu(tf.nn.conv2d(Conv4_1, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        Conv5_1 = None
        with tf.name_scope('Conv5_1'):
            W = self.weight_variable([3, 3, 512, 512]) 
            b = self.bias_variable([512])
            Conv5_1 = tf.nn.relu(tf.nn.conv2d(Conv5, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        Conv6 = None
        with tf.name_scope('Conv6'):
            W = self.weight_variable([3, 3, 512, 1024]) 
            b = self.bias_variable([1024])
            Conv6 = tf.nn.relu(tf.nn.conv2d(Conv5_1, W, strides=[1,2,2,1], padding='SAME') + b)
                     
        Conv6_1 = None
        with tf.name_scope('Conv6_1'):
            W = self.weight_variable([3, 3, 1024, 1024]) 
            b = self.bias_variable([1024])
            Conv6_1 = tf.nn.relu(tf.nn.conv2d(Conv6, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        loss6 = None
        with tf.name_scope('loss6'):
            W = self.weight_variable([3, 3, 1024, 1]) 
            b = self.bias_variable([1])
            loss6 = tf.nn.relu(tf.nn.conv2d(Conv6_1, W, strides=[1,1,1,1], padding='SAME') + b)
              
        ''' deconvolution layers : upconv5(loss5) - upconv4(loss4) - upconv3(loss3) - upconv2(loss2) - upconv1 - loss1 '''
        upconv5 = None
        with tf.name_scope('upconv5'):
            W = self.weight_variable([4, 4, 512, 1024]) 
            b = self.bias_variable([512])
            upconv5 = tf.nn.conv2d_transpose(Conv6_1, W, 
                                             output_shape = [self.batch_size, np.int32(self.model_in_height / 32), np.int32(self.model_in_width / 32), 512], 
                                             strides=[1,2,2,1], padding='SAME') + b
                     
        iconv5 = None
        with tf.name_scope('iconv5'):
            W = self.weight_variable([3, 3, 1025, 512]) 
            b = self.bias_variable([512])
            iconv5 = tf.nn.relu(tf.nn.conv2d(tf.concat([
                tf.image.resize_images(loss6, (np.int32(self.model_in_height / 32), np.int32(self.model_in_width / 32))), 
                upconv5, Conv5_1], axis=3), W, strides=[1,1,1,1], padding='SAME') + b)
                     
        loss5 = None
        with tf.name_scope('loss5'):
            W = self.weight_variable([3, 3, 512, 1]) 
            b = self.bias_variable([1])
            loss5 = tf.nn.relu(tf.nn.conv2d(iconv5, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        upconv4 = None
        with tf.name_scope('upconv4'):
            W = self.weight_variable([4, 4, 256, 512]) 
            b = self.bias_variable([256])
            upconv4 = tf.nn.conv2d_transpose(iconv5, W, 
                                             output_shape = [self.batch_size, np.int32(self.model_in_height / 16), np.int32(self.model_in_width / 16), 256], 
                                             strides=[1,2,2,1], padding='SAME') + b
                     
        iconv4 = None
        with tf.name_scope('iconv4'):
            W = self.weight_variable([3, 3, 769, 256]) 
            b = self.bias_variable([256])
            iconv4 = tf.nn.relu(tf.nn.conv2d(tf.concat([
                tf.image.resize_images(loss5, (np.int32(self.model_in_height / 16), np.int32(self.model_in_width / 16))), 
                upconv4, Conv4_1], axis=3), W, strides=[1,1,1,1], padding='SAME') + b)
                     
        loss4 = None
        with tf.name_scope('loss4'):
            W = self.weight_variable([3, 3, 256, 1]) 
            b = self.bias_variable([1])
            loss4 = tf.nn.relu(tf.nn.conv2d(iconv4, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        upconv3 = None
        with tf.name_scope('upconv3'):
            W = self.weight_variable([4, 4, 128, 256]) 
            b = self.bias_variable([128])
            upconv3 = tf.nn.conv2d_transpose(iconv4, W, 
                                             output_shape = [self.batch_size, np.int32(self.model_in_height / 8), np.int32(self.model_in_width / 8), 128], 
                                             strides=[1,2,2,1], padding='SAME') + b
                     
        iconv3 = None
        with tf.name_scope('iconv3'):
            W = self.weight_variable([3, 3, 385, 128]) 
            b = self.bias_variable([128])
            iconv3 = tf.nn.relu(tf.nn.conv2d(tf.concat([
                tf.image.resize_images(loss4, (np.int32(self.model_in_height / 8), np.int32(self.model_in_width / 8))), 
                upconv3, Conv3_1], axis=3), W, strides=[1,1,1,1], padding='SAME') + b)
                     
        loss3 = None
        with tf.name_scope('loss3'):
            W = self.weight_variable([3, 3, 128, 1]) 
            b = self.bias_variable([1])
            loss3 = tf.nn.relu(tf.nn.conv2d(iconv3, W, strides=[1,1,1,1], padding='SAME') + b)
                     
        upconv2 = None
        with tf.name_scope('upconv2'):
            W = self.weight_variable([4, 4, 64, 128]) 
            b = self.bias_variable([64])
            upconv2 = tf.nn.conv2d_transpose(iconv3, W, 
                                             output_shape = [self.batch_size, np.int32(self.model_in_height / 4), np.int32(self.model_in_width / 4), 64], 
                                             strides=[1,2,2,1], padding='SAME') + b
                     
        iconv2 = None
        with tf.name_scope('iconv2'):
            W = self.weight_variable([3, 3, 193, 64]) 
            b = self.bias_variable([64])
            iconv2 = tf.nn.relu(tf.nn.conv2d(tf.concat([
                tf.image.resize_images(loss3, (np.int32(self.model_in_height / 4), np.int32(self.model_in_width / 4))), 
                upconv2, left_Conv2], axis=3), W, strides=[1,1,1,1], padding='SAME') + b)
                     
        loss2 = None
        with tf.name_scope('loss2'):
            W = self.weight_variable([3, 3, 64, 1]) 
            b = self.bias_variable([1])
            loss2 = tf.nn.relu(tf.nn.conv2d(iconv2, W, strides=[1,1,1,1], padding='SAME') + b)
            
        upconv1 = None
        with tf.name_scope('upconv1'):
            W = self.weight_variable([4, 4, 32, 64]) 
            b = self.bias_variable([32])
            upconv1 = tf.nn.conv2d_transpose(iconv2, W, 
                                             output_shape = [self.batch_size, np.int32(self.model_in_height / 2), np.int32(self.model_in_width / 2), 32], 
                                             strides=[1,2,2,1], padding='SAME') + b
                     
        iconv1 = None
        with tf.name_scope('iconv1'):
            W = self.weight_variable([3, 3, 97, 32]) 
            b = self.bias_variable([32])
            iconv1 = tf.nn.relu(tf.nn.conv2d(tf.concat([
                tf.image.resize_images(loss2, (np.int32(self.model_in_height / 2), np.int32(self.model_in_width / 2))), 
                upconv1, left_Conv1], axis=3), W, strides=[1,1,1,1], padding='SAME') + b)
            
        loss1 = None
        with tf.name_scope('loss1'):
            W = self.weight_variable([3, 3, 32, 1]) 
            b = self.bias_variable([1])
            loss1 = tf.nn.relu(tf.nn.conv2d(iconv1, W, strides=[1,1,1,1], padding='SAME') + b)

        prediction = loss1
        cost = tf.losses.mean_squared_error(ground_truth, prediction)
        training_loss = tf.summary.scalar('training_loss', cost)
        validation_loss = tf.summary.scalar('validation_loss', cost)
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return train_step, cost, prediction, training_loss, validation_loss
    
    def inference(self, left_image , right_image, ground_truth):            
        if self.mode == 'correlation_':
            return self.DispNetCorr_(left_image, right_image, ground_truth)

class FlowNet(object):
    """
    correlation_layer
    https://github.com/jgorgenucsd/corr_tf/blob/master/flownet.py#L59
    """

    def __init__(self, img_height, img_width, img_depth,  learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate

        self.model_in_height = self.resize(img_height)
        self.model_in_width = self.resize(img_width)
        self.model_in_depth = 3

        self.model_out_height = int(self.model_in_height / 4)
        self.model_out_width = int(self.model_in_width / 4)
        self.model_out_depth = 1

        print('input image resized by (height = %s,' %self.model_in_height, 'width = %s)' %self.model_in_width)

    def resize(self, value, multiple = 64):
        n = 1
        condition = False
        diff_before = value
        diff_after = 0

        while not condition:
            diff_after = abs(multiple * n - value)

            if diff_after < diff_before:
                diff_before = diff_after
            else:
                condition = True
                return multiple * (n - 1)
            n += 1

    def FlowNetSimple(self, input): 
        ''' convolution layers : Conv1 - Conv2 - Conv3 - Conv3_1 - Conv4 - Conv4_1 - Conv5 - Conv5_1 - Conv6 '''
        Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='Conv1')(input)
        Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='Conv2')(Conv1)
        Conv3 = Conv2D(256, (5, 5), (2, 2), padding='same', activation='relu', name='Conv3')(Conv2)
        Conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv3_1')(Conv3)
        Conv4 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv4')(Conv3_1)
        Conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv4_1')(Conv4)
        Conv5 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv5')(Conv4_1)
        Conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv5_1')(Conv5)
        Conv6 = Conv2D(1024, (3, 3), (2, 2), padding='same', activation='relu', name='Conv6')(Conv5_1)
        
        ''' deconvolution layers : deconv5(flow5) - deconv4(flow4) - deconv3(flow3) - deconv2 - prediction '''
        deconv5 = Conv2DTranspose(512, (3, 3), (2, 2), padding='same', activation='relu', name='deconv5')(Conv6)
        concat1 = concatenate([deconv5, Conv5_1], axis = 3, name='concat1')

        flow5 = Conv2D(2, (3, 3), padding='same', name='flow5')(concat1)
        flow5_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow5_up')(flow5)

        deconv4 = Conv2DTranspose(256, (3, 3), (2, 2), padding='same', activation='relu', name='deconv4')(concat1)
        concat2 = concatenate([deconv4, Conv4_1, flow5_up], axis = 3, name='concat2')

        flow4 = Conv2D(1, (3, 3), padding='same', name='flow4')(concat2)
        flow4_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow4_up')(flow4)

        deconv3 = Conv2DTranspose(128, (5, 5), (2, 2), padding='same', activation='relu', name='deconv3')(concat2)
        concat3 = concatenate([deconv3, Conv3_1, flow4_up], axis = 3, name='concat3')

        flow3 = Conv2D(1, (3, 3), padding='same', name='flow3')(concat3)
        flow3_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow3_up')(flow3)

        deconv2 = Conv2DTranspose(64, (5, 5), (2, 2), padding='same', activation='relu', name='deconv2')(concat3)
        concat4 = concatenate([deconv2, Conv2, flow3_up], axis = 3, name='concat4')
        prediction = Conv2D(1, (3, 3), padding='same', name='prediction')(concat4)

        return prediction

    def FlowNetCorr(self, input_left, input_right):

        left_Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='left_Conv1')(input_left)
        left_Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='left_Conv2')(left_Conv1)
        left_Conv3 = Conv2D(256, (5, 5), (2, 2), padding='same', activation='relu', name='left_Conv3')(left_Conv2)

        right_Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='right_Conv1')(input_right)
        right_Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='right_Conv2')(right_Conv1)
        right_Conv3 = Conv2D(256, (5, 5), (2, 2), padding='same', activation='relu', name='right_Conv3')(right_Conv2)

        max_disp = 10
        layer_list = []
        dotLayer = Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]), axis = -1, keepdims = True), name = 'dotLayer')
        for i in range(-2 * max_disp, 2 * max_disp + 2, 2):
            for j in range(-2 * max_disp, 2 * max_disp + 2, 2):
                slice_height = int(self.model_in_height / 8) - abs(j)
                slice_width = int(self.model_in_width / 8) - abs(i)
                start_y = abs(j) if j < 0 else 0
                start_x = abs(i) if i < 0 else 0
                top_pad    = j if (j>0) else 0
                bottom_pad = start_y
                left_pad   = i if (i>0) else 0
                right_pad  = start_x
                
                gather_layer = Lambda(lambda x: tf.pad(tf.slice(x, begin=[0, start_y, start_x,0], size=[-1, slice_height, slice_width, -1]),
                                                        paddings=[[0,0], [top_pad,bottom_pad], [left_pad,right_pad], [0,0]]),
                                        name='gather_{}_{}'.format(i, j))(right_Conv3)
                current_layer = dotLayer([left_Conv3,gather_layer])
                layer_list.append(current_layer)
        Corr_441 = Lambda(lambda x: tf.concat(x, 3),name='Corr_441')(layer_list)
        Conv_redir = Conv2D(32, (1, 1), (1, 1), padding='same', activation='relu', name='Conv_redir')(left_Conv3)
        Corr = concatenate([Corr_441, Conv_redir], axis = 3, name='Corr')

        Conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv3_1')(Corr)
        Conv4 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv4')(Conv3_1)
        Conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv4_1')(Conv4)
        Conv5 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv5')(Conv4_1)
        Conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv5_1')(Conv5)
        Conv6 = Conv2D(1024, (3, 3), (2, 2), padding='same', activation='relu', name='Conv6')(Conv5_1)
        
        ''' deconvolution layers : deconv5(flow5) - deconv4(flow4) - deconv3(flow3) - deconv2 - prediction '''
        deconv5 = Conv2DTranspose(512, (3, 3), (2, 2), padding='same', activation='relu', name='deconv5')(Conv6)
        concat1 = concatenate([deconv5, Conv5_1], axis = 3, name='concat1')

        flow5 = Conv2D(2, (3, 3), padding='same', name='flow5')(concat1)
        flow5_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow5_up')(flow5)

        deconv4 = Conv2DTranspose(256, (3, 3), (2, 2), padding='same', activation='relu', name='deconv4')(concat1)
        concat2 = concatenate([deconv4, Conv4_1, flow5_up], axis = 3, name='concat2')

        flow4 = Conv2D(1, (3, 3), padding='same', name='flow4')(concat2)
        flow4_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow4_up')(flow4)

        deconv3 = Conv2DTranspose(128, (5, 5), (2, 2), padding='same', activation='relu', name='deconv3')(concat2)
        concat3 = concatenate([deconv3, Conv3_1, flow4_up], axis = 3, name='concat3')

        flow3 = Conv2D(1, (3, 3), padding='same', name='flow3')(concat3)
        flow3_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='flow3_up')(flow3)

        deconv2 = Conv2DTranspose(64, (5, 5), (2, 2), padding='same', activation='relu', name='deconv2')(concat3)
        concat4 = concatenate([deconv2, left_Conv2, flow3_up], axis = 3, name='concat4')
        prediction = Conv2D(1, (3, 3), padding='same', name='prediction')(concat4)

        return prediction

    def inference(self, mode = 'simple'):

        left_image = Input(shape=(self.model_in_height, self.model_in_width, self.model_in_depth), name='left_input')
        right_image = Input(shape=(self.model_in_height, self.model_in_width, self.model_in_depth), name='right_image')

        if mode == 'simple':
            concate_view = concatenate([left_image, right_image], axis = 3, name='concate_view')
            prediction = self.FlowNetSimple(concate_view)

            FlowNet = Model(inputs = [left_image, right_image], outputs = [prediction])
            opt = Adam(lr=self.learning_rate)
            FlowNet.compile(optimizer=opt, loss='mae')
            FlowNet.summary() 
            
            return FlowNet

        if mode == 'correlation':
            prediction = self.FlowNetCorr(left_image, right_image)

            FlowNet = Model(inputs = [left_image, right_image], outputs = [prediction])
            opt = Adam(lr=self.learning_rate)
            FlowNet.compile(optimizer=opt, loss='mae')
            FlowNet.summary() 
            
            return FlowNet

class EPINet(object):

    def __init__(self, img_height, img_width, view_n, conv_depth, filt_num, learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.view_n = view_n
        self.conv_depth = conv_depth
        self.filt_num = filt_num
        self.learning_rate = learning_rate

    def layer1_multistream(self, input_dim1, input_dim2, input_dim3, filt_num):    
        seq = Sequential()
        ''' Multi-Stream layer : Conv - Relu - Conv - BN - Relu  '''

        # (Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
        for i in range(3):
            (Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
            (Activation('relu', name='S1_relu1%d' %(i))) 
            (Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) )) 
            (BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
            (Activation('relu', name='S1_relu2%d' %(i))) 

        (Reshape((input_dim1-6,input_dim2-6,int(filt_num))))

        return seq  

    def layer2_merged(self, input_dim1, input_dim2, input_dim3, filt_num, conv_depth):
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        
        seq = Sequential()
        
        for i in range(conv_depth):
            (Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
            (Activation('relu', name='S2_relu1%d' %(i))) 
            (Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) 
            (BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
            (Activation('relu', name='S2_relu2%d' %(i)))
            
        return seq     

    def layer3_last(self, input_dim1, input_dim2, input_dim3, filt_num):   
        ''' last layer : Conv - Relu - Conv ''' 
        
        seq = Sequential()
        
        for i in range(1):
            (Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
            (Activation('relu', name='S3_relu1%d' %(i)))
            
        (Conv2D(1,(2,2), padding='valid', name='S3_last')) 

        return seq 

    def inference(self):

        ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
        input_stack_90d = Input(shape=(self.img_height,self.img_width,len(self.view_n)), name='input_stack_90d')
        input_stack_0d= Input(shape=(self.img_height,self.img_width,len(self.view_n)), name='input_stack_0d')
        input_stack_45d= Input(shape=(self.img_height,self.img_width,len(self.view_n)), name='input_stack_45d')
        input_stack_M45d= Input(shape=(self.img_height,self.img_width,len(self.view_n)), name='input_stack_M45d')
        
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        mid_90d=self.layer1_multistream(self.img_height,self.img_width,len(self.view_n),int(self.filt_num))(input_stack_90d)
        mid_0d=self.layer1_multistream(self.img_height,self.img_width,len(self.view_n),int(self.filt_num))(input_stack_0d)    
        mid_45d=self.layer1_multistream(self.img_height,self.img_width,len(self.view_n),int(self.filt_num))(input_stack_45d)    
        mid_M45d=self.layer1_multistream(self.img_height,self.img_width,len(self.view_n),int(self.filt_num))(input_stack_M45d)   

        ''' Merge layers ''' 
        mid_merged = concatenate([mid_90d,mid_0d,mid_45d,mid_M45d],  name='mid_merged')
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_ = self.layer2_merged(self.img_height-6,self.img_width-6,int(4*self.filt_num),int(4*self.filt_num),self.conv_depth)(mid_merged)

        ''' Last Conv layer : Conv - Relu - Conv '''
        output = self.layer3_last(self.img_height-18,self.img_width-18,int(4*self.filt_num),int(4*self.filt_num))(mid_merged_)

        epinet = Model(inputs = [input_stack_90d,input_stack_0d,
                                input_stack_45d,input_stack_M45d], outputs = [output])
        opt = RMSprop(lr=self.learning_rate)
        epinet.compile(optimizer=opt, loss='mae')
        epinet.summary() 
        
        return epinet
        