# tensorflow keras
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape, Conv2DTranspose
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate, MaxPooling2D
from tensorflow.contrib.keras.api.keras.backend import resize_images

# class FlowNetCorr(object):

class FlowNet(object):

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
        #deconv5 = conv2d_transpose(Conv6, (3, 3), tf.shape(Conv5_1), (2, 2), padding='same')
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
        