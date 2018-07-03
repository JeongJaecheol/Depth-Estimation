# tensorflow keras
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape, Conv2DTranspose
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate, MaxPooling2D
from tensorflow.contrib.keras.api.keras.backend import resize_images

# class FlowNetCorr(object):

class FlowNetSimple(object):

    def __init__(self, img_height, img_width, img_depth,  learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate

    def conv_layers(self):
        seq = Sequential()
        ''' convolution layers : Conv1 - Conv2 - Conv3 - Conv3_1 - Conv4 - Conv4_1 - Conv5 - Conv5_1 - Conv6 '''
        seq.add(Conv2D(64, (7, 7), input_shape=(self.img_height, self.img_width, self.img_depth), padding='same', name='Conv1'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Conv2D(128, (5, 5), padding='same', name='Conv2'))
        Conv2 = seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Conv2D(256, (5, 5), padding='same', name='Conv3'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        Conv3_1 = seq.add(Conv2D(256, (3, 3), padding='same', name='Conv3_1'))
        seq.add(Conv2D(512, (3, 3), padding='same', name='Conv4'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        Conv4_1 = seq.add(Conv2D(512, (3, 3), padding='same', name='Conv4_1'))
        seq.add(Conv2D(512, (3, 3), padding='same', name='Conv5'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        Conv5_1 = seq.add(Conv2D(512, (3, 3), padding='same', name='Conv5_1'))
        seq.add(Conv2D(1024, (3, 3), padding='same', name='Conv6'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))

        return seq, Conv5_1, Conv4_1, Conv3_1, Conv2

    def deconv_layers(self, Conv5_1, Conv4_1, Conv3_1, Conv2): 
        seq = Sequential()
        ''' deconvolution layers : deconv5(flow5) - deconv4(flow4) - deconv3(flow3) - deconv2 - prediction '''
        deconv5 = seq.add(Conv2DTranspose(512, (5, 5), padding='same', name='deconv5'))
        seq.add(concatenate([deconv5, Conv5_1]))
        flow5 = seq.add(Conv2D(1, (5, 5), padding='same', name='flow5'))

        deconv4 = seq.add(Conv2DTranspose(512, (5, 5), padding='same', name='deconv4'))
        seq.add(concatenate([deconv4, Conv4_1, flow5]))
        flow4 = seq.add(Conv2D(1, (5, 5), padding='same', name='flow4'))

        deconv3 = seq.add(Conv2DTranspose(512, (5, 5), padding='same', name='deconv3'))
        seq.add(concatenate([deconv3, Conv3_1, flow4]))
        flow3 = seq.add(Conv2D(1, (5, 5), padding='same', name='flow3'))

        deconv2 = seq.add(Conv2DTranspose(512, (5, 5), padding='same', name='deconv2'))
        seq.add(concatenate([deconv2, Conv2, flow3]))
        prediction = seq.add(Conv2D(1, (5, 5), padding='same', name='prediction'))

        return prediction

    def inference(self):

        stack2image = Input(shape=(self.img_height, self.img_width, self.img_width * 2), name='stack2image')
        encoded, Conv5_1, Conv4_1, Conv3_1, Conv2 = self.conv_layers()(stack2image)
        prediction = self.deconv_layers(Conv5_1, Conv4_1, Conv3_1, Conv2)(encoded)

        FlowNetS = Model(inputs = [stack2image], outputs = [prediction])
        opt = Adam(lr=self.learning_rate)
        FlowNetS.compile(optimizer=opt, loss='mae')
        FlowNetS.summary() 
        
        return FlowNetS

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

        # seq.add(Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
        for i in range(3):
            seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
            seq.add(Activation('relu', name='S1_relu1%d' %(i))) 
            seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) )) 
            seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
            seq.add(Activation('relu', name='S1_relu2%d' %(i))) 

        seq.add(Reshape((input_dim1-6,input_dim2-6,int(filt_num))))

        return seq  

    def layer2_merged(self, input_dim1, input_dim2, input_dim3, filt_num, conv_depth):
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        
        seq = Sequential()
        
        for i in range(conv_depth):
            seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
            seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
            seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) 
            seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
            seq.add(Activation('relu', name='S2_relu2%d' %(i)))
            
        return seq     

    def layer3_last(self, input_dim1, input_dim2, input_dim3, filt_num):   
        ''' last layer : Conv - Relu - Conv ''' 
        
        seq = Sequential()
        
        for i in range(1):
            seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
            seq.add(Activation('relu', name='S3_relu1%d' %(i)))
            
        seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last')) 

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
        