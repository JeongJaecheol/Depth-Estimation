# tensorflow keras
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation, Lambda
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape, Conv2DTranspose
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate, MaxPooling2D
from tensorflow.contrib.keras.api.keras.backend import resize_images

class DispNet(object):
    """
    original tensorflow code
    https://github.com/ZhijianJiang/DispNet-TensorFlow/blob/master/DispNet.py

    correlation_map
    https://github.com/fedor-chervinskii/dispflownet-tf/blob/master/dispnet.py
    """

    def __init__(self, img_height, img_width, img_depth,  learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate

        self.model_in_height = self.resize(img_height)
        self.model_in_width = self.resize(img_width)
        self.model_in_depth = 3

        self.model_out_height = int(self.model_in_height / 2)
        self.model_out_width = int(self.model_in_width / 2)
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

    def DispNetSimple(self, input): 
        ''' convolution layers : Conv1 - Conv2 - Conv3 - Conv3_1 - Conv4 - Conv4_1 - Conv5 - Conv5_1 - Conv6 - Conv6_1 '''
        Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='Conv1')(input)
        Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='Conv2')(Conv1)
        Conv3 = Conv2D(256, (5, 5), (2, 2), padding='same', activation='relu', name='Conv3')(Conv2)
        Conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv3_1')(Conv3)
        Conv4 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv4')(Conv3_1)
        Conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv4_1')(Conv4)
        Conv5 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv5')(Conv4_1)
        Conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv5_1')(Conv5)
        Conv6 = Conv2D(1024, (3, 3), (2, 2), padding='same', activation='relu', name='Conv6')(Conv5_1)
        Conv6_1 = Conv2D(1024, (3, 3), padding='same', activation='relu', name='Conv6_1')(Conv6)
        loss6 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss6')(Conv6_1)
        loss6_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss6_up')(loss6)
        
        ''' deconvolution layers : upconv5(loss5) - upconv4(loss4) - upconv3(loss3) - upconv2(loss2) - upconv1 - loss1 '''
        upconv5 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv5')(Conv6_1)
        concat1 = concatenate([upconv5, loss6_up, Conv5_1], axis = 3, name='concat1')
        iconv5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv5')(concat1)
        loss5 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss5')(iconv5)
        loss5_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss5_up')(loss5)

        upconv4 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv4')(iconv5)
        concat2 = concatenate([upconv4, loss5_up, Conv4_1], axis = 3, name='concat2')
        iconv4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv4')(concat2)
        loss4 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss4')(iconv4)
        loss4_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss4_up')(loss4)

        upconv3 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv3')(iconv4)
        concat3 = concatenate([upconv3, loss4_up, Conv3_1], axis = 3, name='concat3')
        iconv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv3')(concat3)
        loss3 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss3')(iconv3)
        loss3_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss3_up')(loss3)

        upconv2 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv2')(iconv3)
        concat4 = concatenate([upconv2, loss3_up, Conv2], axis = 3, name='concat4')
        iconv2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv2')(concat4)
        loss2 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss2')(iconv2)
        loss2_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss2_up')(loss2)

        upconv1 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv1')(iconv2)
        concat5 = concatenate([upconv1, loss2_up, Conv1], axis = 3, name='concat5')
        iconv1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv1')(concat5)
        loss1 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss1')(iconv1)

        loss_list = [loss1, loss2, loss3, loss4, loss5, loss6]

        return loss_list

    def DispNetCorr(self, input_left, input_right):

        left_Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='left_Conv1')(input_left)
        left_Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='left_Conv2')(left_Conv1)

        right_Conv1 = Conv2D(64, (7, 7), (2, 2), padding='same', activation='relu', name='right_Conv1')(input_right)
        right_Conv2 = Conv2D(128, (5, 5), (2, 2), padding='same', activation='relu', name='right_Conv2')(right_Conv1)

        max_disp = 40
        corr_tensors = []
        for i in range(-max_disp, 0, 1):
            shifted = Lambda(lambda x: tf.pad(tf.slice(x, [0]*4, [-1, -1, x.shape[2].value + i, -1]),
                            [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT"), name='shifted_{}_0'.format(i))(right_Conv2)
            corr = Lambda(lambda x: tf.reduce_mean(tf.multiply(x[0], x[1]), axis=3), name='reduce_mean_{}_0'.format(i))([shifted, left_Conv2])
            corr_tensors.append(corr)
        for i in range(max_disp + 1):
            shifted = Lambda(lambda x: tf.pad(tf.slice(x, [0, 0, i, 0], [-1]*4),
                            [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT"), name='shifted_0_{}'.format(i))(left_Conv2)
            corr = Lambda(lambda x: tf.reduce_mean(tf.multiply(x[0], x[1]), axis=3), name='reduce_mean_0_{}'.format(i))([shifted, right_Conv2])
            corr_tensors.append(corr)
        Corr_81 = Lambda(lambda x: tf.transpose(tf.stack(x), perm=[1, 2, 3, 0]), name='Corr_81')(corr_tensors)
        Conv_redir = Conv2D(64, (1, 1), (1, 1), padding='same', activation='relu', name='Conv_redir')(left_Conv2)
        Corr = concatenate([Corr_81, Conv_redir], axis = 3, name='Corr')

        Conv3 = Conv2D(256, (5, 5), (2, 2), padding='same', activation='relu', name='Conv3')(Corr)
        Conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv3_1')(Conv3)
        Conv4 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv4')(Conv3_1)
        Conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv4_1')(Conv4)
        Conv5 = Conv2D(512, (3, 3), (2, 2), padding='same', activation='relu', name='Conv5')(Conv4_1)
        Conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv5_1')(Conv5)
        Conv6 = Conv2D(1024, (3, 3), (2, 2), padding='same', activation='relu', name='Conv6')(Conv5_1)
        Conv6_1 = Conv2D(1024, (3, 3), padding='same', activation='relu', name='Conv6_1')(Conv6)
        loss6 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss6')(Conv6_1)
        loss6_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss6_up')(loss6)
        
        ''' deconvolution layers : upconv5(loss5) - upconv4(loss4) - upconv3(loss3) - upconv2(loss2) - upconv1 - loss1 '''
        upconv5 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv5')(Conv6_1)
        concat1 = concatenate([upconv5, loss6_up, Conv5_1], axis = 3, name='concat1')
        iconv5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv5')(concat1)
        loss5 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss5')(iconv5)
        loss5_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss5_up')(loss5)

        upconv4 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv4')(iconv5)
        concat2 = concatenate([upconv4, loss5_up, Conv4_1], axis = 3, name='concat2')
        iconv4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv4')(concat2)
        loss4 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss4')(iconv4)
        loss4_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss4_up')(loss4)

        upconv3 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv3')(iconv4)
        concat3 = concatenate([upconv3, loss4_up, Conv3_1], axis = 3, name='concat3')
        iconv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv3')(concat3)
        loss3 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss3')(iconv3)
        loss3_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss3_up')(loss3)

        upconv2 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv2')(iconv3)
        concat4 = concatenate([upconv2, loss3_up, left_Conv2], axis = 3, name='concat4')
        iconv2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv2')(concat4)
        loss2 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss2')(iconv2)
        loss2_up = Conv2DTranspose(2, (3, 3), (2, 2), padding='same', name='loss2_up')(loss2)

        upconv1 = Conv2DTranspose(512, (4, 4), (2, 2), padding='same', activation='relu', name='upconv1')(iconv2)
        concat5 = concatenate([upconv1, loss2_up, left_Conv1], axis = 3, name='concat5')
        iconv1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='iconv1')(concat5)
        loss1 = Conv2D(1, (3, 3), padding='same', activation='relu', name='loss1')(iconv1)

        loss_list = [loss1, loss2, loss3, loss4, loss5, loss6]

        return loss_list

    def inference(self, mode = 'simple'):

        left_image = Input(shape=(self.model_in_height, self.model_in_width, self.model_in_depth), name='left_input')
        right_image = Input(shape=(self.model_in_height, self.model_in_width, self.model_in_depth), name='right_image')

        if mode == 'simple':
            concate_view = concatenate([left_image, right_image], axis = 3, name='concate_view')
            loss_list = self.DispNetSimple(concate_view)

            DispNet = Model(inputs = [left_image, right_image], outputs = loss_list)
            opt = Adam(lr=self.learning_rate)
            DispNet.compile(optimizer=opt, loss='mae', loss_weights=[1/2, 1/4, 1/8, 1/16, 1/32, 1/32])
            DispNet.summary() 
            
            return DispNet

        if mode == 'correlation':
            loss_list = self.DispNetCorr(left_image, right_image)

            DispNet = Model(inputs = [left_image, right_image], outputs = loss_list)
            opt = Adam(lr=self.learning_rate)
            DispNet.compile(optimizer=opt, loss='mae', loss_weights=[1/2, 1/4, 1/8, 1/16, 1/32, 1/32])
            DispNet.summary() 
            
            return DispNet

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
        