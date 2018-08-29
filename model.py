# tensorflow keras
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Reshape, Permute, Lambda
from tensorflow.python.keras.layers import Add, add, multiply
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.python.keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.python.keras.layers import concatenate, MaxPooling2D
from tensorflow.python.keras.optimizers import RMSprop, Adam

class GCNet(object):
    def __init__(self, img_height, img_width, img_depth, disp_range, learning_rate, num_of_gpu = 1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.disp_range = disp_range
        self.learning_rate = learning_rate
        self.num_of_gpu = num_of_gpu

        self.model_in_height = self.resize(img_height)
        self.model_in_width = self.resize(img_width)
        self.model_in_depth = 3

        print('input image resized by (height = %s,' %self.model_in_height, 'width = %s)' %self.model_in_width)

    def resize(self, value, multiple = 32):
        n = 1
        condition = False

        while not condition:
            if (multiple * n) <= value:
                n += 1
            else:
                condition = True
                return multiple * (n - 1)

    def SoftArgMax(self, x, height, width, disp_range):
        tmp = tf.squeeze(x, squeeze_dims=-1)
        softmax = tf.nn.softmax(-tmp)
        disp_mat = tf.constant(list(map(lambda x: x, range(1, 192+1, 1))), shape=(192, 512, 960))
        disp_mat = tf.cast(disp_mat, tf.float32)
        result = tf.multiply(softmax, disp_mat)
        result = tf.reduce_sum(result, axis = 1)
        return result

    def CostVolume(self, inputs, max_d):
        left_tensor, right_tensor = inputs
        shape = right_tensor.shape
        right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
        disparity_costs = []
        for d in reversed(range(max_d)):
            left_tensor_slice = left_tensor
            right_tensor_slice = tf.slice(right_tensor, begin = [0, 0, d, 0], size = [-1, -1, shape[2], -1])
            cost = K.concatenate([left_tensor_slice, right_tensor_slice], axis = 3)
            disparity_costs.append(cost)
        cost_volume = K.stack(disparity_costs, axis = 1)
        return cost_volume

    def Unaryfeatures(self, input):
        layer1 = Conv2D(32, (5, 5), (2, 2), padding='same', activation='relu')(input)
        layer2 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer1)
        layer3 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer2)        
        short_cut1 = add([layer1, layer3])

        layer4 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut1)
        layer5 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer4)
        short_cut2 = add([layer4, layer5])
        layer6 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut2)
        layer7 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer6)
        short_cut3 = add([layer6, layer7])
        layer8 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut3)
        layer9 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer8)
        short_cut4 = add([layer8, layer9])
        layer10 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut4)
        layer11 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer10)
        short_cut5 = add([layer10, layer11])
        layer12 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut5)
        layer13 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer12)
        short_cut6 = add([layer12, layer13])
        layer14 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut6)
        layer15 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer14)
        short_cut7 = add([layer14, layer15])
        layer16 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(short_cut7)
        layer17 = Conv2D(32, (3, 3), (1, 1), padding='same', activation='relu')(layer16)
        short_cut8 = add([layer16, layer17])
        
        layer18 = Conv2D(32, (3, 3), (1, 1), padding='same')(short_cut8)

        return layer18

    def LearningRegularization(self, input):
        layer19 = Conv3D(32, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(input)
        layer20 = Conv3D(32, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer19)
        layer21 = Conv3D(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(layer20)
        layer22 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer21)
        layer23 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer22)
        layer24 = Conv3D(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(layer23)
        layer25 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer24)
        layer26 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer25)
        layer27 = Conv3D(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(layer26)
        layer28 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer27)
        layer29 = Conv3D(64, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer28)
        layer30 = Conv3D(128, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(layer29)
        layer31 = Conv3D(128, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer30)
        layer32 = Conv3D(128, (3, 3, 3), (1, 1, 1), padding='same', activation='relu')(layer31)

        layer33 = Conv3DTranspose(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(layer32)
        short_cut1 = add([layer29, layer33])
        layer34 = Conv3DTranspose(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(short_cut1)
        short_cut2 = add([layer26, layer34])
        layer35 = Conv3DTranspose(64, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(short_cut2)
        short_cut3 = add([layer23, layer35])
        layer36 = Conv3DTranspose(32, (3, 3, 3), (2, 2, 2), padding='same', activation='relu')(short_cut3)
        short_cut4 = add([layer20, layer36])
        layer37 = Conv3DTranspose(1, (3, 3, 3), (2, 2, 2), padding='same')(short_cut4)

        return Lambda(self.SoftArgMax, 
                        arguments = {'height':self.model_in_height, 'width':self.model_in_width, 'disp_range':self.disp_range}, 
                        )(layer37)

    def inference(self):
        input_shape = (self.model_in_height, self.model_in_width, self.img_depth)
        l_img = Input(shape = input_shape, dtype = "float32", name='l_img')
        r_img = Input(shape = input_shape, dtype = "float32", name='r_img')
        
        input_img = Input(shape = input_shape, dtype = "float32", name='l_img')
        feature = self.Unaryfeatures(input_img)
        Unaryfeatures = Model(inputs = input_img, outputs = feature)

        l_feature = Unaryfeatures(l_img)
        r_feature = Unaryfeatures(r_img) 

        unifeatures = [l_feature, r_feature]   
        cv_l = Lambda(self.CostVolume, arguments = {'max_d':(int)(self.disp_range / 2)}, name='CostVolume_left')(unifeatures)  
        unifeatures = [r_feature, l_feature]
        cv_r = Lambda(self.CostVolume, arguments = {'max_d':(int)(self.disp_range / 2)}, name='CostVolume_right')(unifeatures)
        
        input_costVolume = Input(shape = ((int)(self.disp_range / 2), (int)(self.model_in_height / 2), (int)(self.model_in_width / 2), 64), dtype = "float32", name='l_img')
        disp_map = self.LearningRegularization(input_costVolume) 
        LearningRegularization = Model(inputs = input_costVolume, outputs = disp_map) 

        l_disp_map = LearningRegularization(cv_l)
        r_disp_map = LearningRegularization(cv_r)

        GCNet = Model(inputs = [l_img , r_img], outputs = [l_disp_map, r_disp_map])

        opt = RMSprop(lr = self.learning_rate, rho = 0.9, epsilon = 0.00000001, decay = 0.0)
        GCNet.compile(optimizer = opt, loss = "mean_absolute_error")
        if (self.num_of_gpu > 1):
            GCNet = multi_gpu_model(GCNet, gpus = self.num_of_gpu)
        GCNet.summary() 

        return GCNet

class DispNet(object):
    """
    original tensorflow code
    https://github.com/ZhijianJiang/DispNet-TensorFlow/blob/master/DispNet.py

    correlation_map
    https://github.com/fedor-chervinskii/dispflownet-tf/blob/master/dispnet.py
    """

    def __init__(self, img_height, img_width, img_depth,  learning_rate, num_of_gpu=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate
        self.num_of_gpu = num_of_gpu

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
        for i in range(max_disp, 0, -1):
            shifted = Lambda(lambda x: tf.pad(tf.slice(x, [0]*4, [-1, -1, x.shape[2].value - i, -1]),
                            [[0, 0], [0, 0], [i, 0], [0, 0]], "CONSTANT"), name='shifted_{}_0'.format(-i))(right_Conv2)
            corr = Lambda(lambda x: tf.reduce_mean(tf.multiply(x[0], x[1]), axis=3), name='reduce_mean_{}_0'.format(-i))([shifted, left_Conv2])
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
            if (self.num_of_gpu > 1):
                DispNet = multi_gpu_model(DispNet, gpus = self.num_of_gpu)
            DispNet.summary() 
            
            return DispNet

        if mode == 'correlation':
            loss_list = self.DispNetCorr(left_image, right_image)

            DispNet = Model(inputs = [left_image, right_image], outputs = loss_list)
            #DispNet = multi_gpu_model(DispNet, gpus=2)
            opt = Adam(lr=self.learning_rate)
            DispNet.compile(optimizer=opt, loss='mae', loss_weights=[1/2, 1/4, 1/8, 1/16, 1/32, 1/32])
            if (self.num_of_gpu > 1):
                DispNet = multi_gpu_model(DispNet, gpus = self.num_of_gpu)
            DispNet.summary() 
            
            return DispNet

class FlowNet(object):
    """
    correlation_layer
    https://github.com/jgorgenucsd/corr_tf/blob/master/flownet.py#L59
    """

    def __init__(self, img_height, img_width, img_depth,  learning_rate, num_of_gpu=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate
        self.num_of_gpu = num_of_gpu

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

        def get_padded_stride(b,displacement_x,displacement_y,height_8,width_8):
            slice_height = (int)(height_8 - abs(displacement_y))
            slice_width = (int)(width_8 - abs(displacement_x))
            start_y = abs(displacement_y) if displacement_y < 0 else 0
            start_x = abs(displacement_x) if displacement_x < 0 else 0
            top_pad    = displacement_y if (displacement_y>0) else 0
            bottom_pad = start_y
            left_pad   = displacement_x if (displacement_x>0) else 0
            right_pad  = start_x
            
            gather_layer = Lambda(lambda x: tf.pad(tf.slice(x,begin=[0,start_y,start_x,0],size=[-1,slice_height,slice_width,-1]),
                                                            paddings=[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]),
                                                    name='gather_{}_{}'.format(displacement_x,displacement_y))(b)
            return gather_layer

        def correlation_layer(conv3_pool_l,conv3_pool_r,max_displacement=20,stride2=2,height_8=self.model_in_height/8,width_8=self.model_in_width/8):
            layer_list = []
            dotLayer = Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]),axis=-1,keep_dims=True),name = 'Dot')
            for i in range(-max_displacement, max_displacement+stride2,stride2):
                for j in range(-max_displacement, max_displacement+stride2,stride2):
                    slice_b = get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
                    current_layer = dotLayer([conv3_pool_l,slice_b])
                    layer_list.append(current_layer)
            return Lambda(lambda x: tf.concat(x, 3),name='441_output_concatenation')(layer_list)

        Corr_441 = correlation_layer(left_Conv3, right_Conv3)
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
            if (self.num_of_gpu > 1):
                FlowNet = multi_gpu_model(FlowNet, gpus = self.num_of_gpu)
            FlowNet.summary() 
            
            return FlowNet

        if mode == 'correlation':
            prediction = self.FlowNetCorr(left_image, right_image)

            FlowNet = Model(inputs = [left_image, right_image], outputs = [prediction])
            opt = Adam(lr=self.learning_rate)
            FlowNet.compile(optimizer=opt, loss='mae')
            if (self.num_of_gpu > 1):
                FlowNet = multi_gpu_model(FlowNet, gpus = self.num_of_gpu)
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
        