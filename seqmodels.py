
import keras
import h5py
from keras import regularizers
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Input
from keras import backend as K
from keras import Model
import numpy as np
from random import choice
from keras import losses
import tensorflow as tf

def MSD_block(I,suf="",n_f=128,filter_size=3,dilation_rates=[1,2,4],data_format="channels_last"):

    f = (filter_size,1)
    d1, d2, d3 = dilation_rates

    conv1a = Conv2D(n_f, f, padding="same", activation="relu", dilation_rate=d1, data_format=data_format, name="conv1a"+suf)(I)
    conv1b = Conv2D(n_f, f, padding="same", activation=None, data_format=data_format, name="conv1b"+suf)(conv1a)
    
    conv2a = Conv2D(in_f, f, padding="same", activation="relu", dilation_rate=d2, data_format=data_format, name="conv2a"+suf)(I)
    conv2b = Conv2D(n_f, f, padding="same", activation=None, dilation_rate=d2, data_format=data_format,name="conv2b"+suf)(conv2a)

    conv3a = Conv2D(n_f, f, padding="same", activation="relu", dilation_rate=d3, data_format=data_format,name="conv3a"+suf)(I)
    conv3b = Conv2D(n_f, f, padding="same", activation=None, dilation_rate=d3, data_format=data_format,name="conv3b"+suf)(conv3a)

    concat1 = Concatenate(name="concat1"+suf)([conv1b, conv2b, conv3b])
    
    return(concat1)

def MSD_like(seqlen=1000, n_features=1000, reg=0.01, dil_1=[1,2,4], dil_2=[8,16,32],data_format="channels_last"):
    x = Input(batch_shape=(None,seqlen,1,4))

    print("Input shape: {0}".format(x.shape))

    block1 = MSD_block(x, dilation_rates=dil_1, filter_size=3 , n_f=128, data_format=data_format, suf="_b1")
    a1 = Activation("relu")(block1)
    print("Block 1 shape: {}".format(a1.shape))

    block2 = MSD_block(a1, dilation_rates=dil_2, filter_size=3, n_f=128, data_format=data_format, suf="_b2")
    print("Block 2 shape: {}".format(block2.shape))

    c1 = Concatenate()([block1, block2])
    print("Concat 1 shape: {}".format(c1.shape))

    c1 = Activation("relu")(c1)
    c1 = Conv2D(64, 1, padding="valid", activation="relu")(c1)
    print("1by1 Conv shape: {}".format(c1.shape))

    bottle1 = Flatten()(c1)
    bottle1 = Dense(32,activation='tanh')(bottle1)
    print("Bottle shape: {}".format(bottle1.shape))

    r = regularizers.l2(reg)
    up1 = Dense(64, activation="relu", kernel_regularizer=r, bias_regularizer=r, name="up1")(bottle1)
    print("Up1 shape: {}".format(up1.shape))
    up2 = Dense(128, activation="relu", kernel_regularizer=r, bias_regularizer=r, name="up2")(up1)
    print("Up2 shape: {}".format(up2.shape))
    up3 = Dense(256, activation="relu", kernel_regularizer=r, bias_regularizer=r, name="up3")(up2)
    print("Up3 shape: {}".format(up3.shape))
    up4 = Dense(n_features, activation="linear", kernel_regularizer=r,bias_regularizer=r, name="up4")(up3)
    print("Up4 shape: {}".format(up4.shape))

    model = Model(inputs=x, outputs=up4)
    return model

def pwm_simple(seqlen=1000, n_features=1000, reg=0, n_f1=128, len_f1=3, data_format="channels_last", mtype=1):
    x = Input(batch_shape=(None,seqlen,1,4))
    print("Input shape: {0}".format(x.shape))
    f = (len_f1, 1)
    conv1 = Conv2D(n_f1, f, activation="relu", padding="same")(x)
    print("Conv1 shape: {}".format(conv1.shape))
    conv2 = Conv2D(n_f1, (7, 1) , activation="relu", padding="same")(conv1)
    print("Conv2 shape: {}".format(conv2.shape))
    pool1 = MaxPooling2D(pool_size=(2, 1),padding='valid')(conv2)
    print("Pool1 shape: {}".format(pool1.shape))
    conv3 = Conv2D(256, (3,1), activation="relu", padding="same")(pool1)
    print("Conv3 shape: {}".format(conv3.shape))
    pool2 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv3)
    print("Pool2 shape: {}".format(pool2.shape))
    conv4 = Conv2D(512, (3,1), activation="relu", padding="same")(pool2)
    print("Conv4 shape: {}".format(conv4.shape))
    pool3 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv4)
    print("Pool3 shape: {}".format(pool3.shape))    
    # regularizer is applied to the last layers, if needed
    r = regularizers.l2(reg)
    flat = Flatten()(conv4)
    print("Flat shape: {}".format(flat.shape))
    dense1 = Dense(128, activation = "linear", kernel_regularizer = r, bias_regularizer = r, name="bottleneck")(flat)
    dense1 = Activation("relu")(dense1)     
    print("Dense1 shape: {}".format(dense1.shape))
    dense2 = Dense(n_features, activation="linear", kernel_regularizer=r, bias_regularizer=r)(dense1)
    print("Dense2 shape: {}".format(dense2.shape))
    if mtype == 1:
        model = Model(inputs=x, outputs=dense2)
        print("Output shape: {}".format(dense2.shape))
        return model
    elif mtype == 2:
        dense2b = Dense(n_features, activation="sigmoid", kernel_regularizer=r, bias_regularizer=r)(dense1)
        print("Dense2b shape: {}".format(dense2.shape))
        out = Concatenate(axis=1)([dense2b, dense2])
        print("Output shape: {}".format(out.shape))
        model = Model(inputs=x, outputs=out)
        return model
    else:
        raise NotImplementedError("mtype {} not implemented.".format(mtype))

def reverse_complement(onehotseq):
    return onehotseq[:,::-1,:,::-1]

def seqloss1(y_true, y_pred):
    w = 0.5
    y_pred_prob, y_pred_value = tf.split(y_pred,2,axis=-1)
    y_true_prob, y_true_value = tf.split(y_true,2,axis=-1)
    err = y_true_value - y_pred_value
    z = tf.zeros_like(err)
    err = tf.where(tf.cast(y_true_prob,bool),z,err)
    sqerr = tf.reduce_sum(tf.square(err))
    crossentrop = losses.binary_crossentropy(y_true_prob, y_pred_prob)
    return (1-w)*crossentrop + w*sqerr

class DataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras
    '''
    def __init__(self, chunks, chunksize, h5file, x_dataset, y_dataset, batch_size=32, chunk_batch_size=10, seqlen=1000,
                n_features=None, shuffle=True, rc=False, y_type=1):
        self.chunks = chunks
        self.chunksize = chunksize
        self.h5file = h5file
        with h5py.File(h5file,'r') as f:
            self.h5len = f[y_dataset].shape[0]
        self.last_chunk = self.h5len // chunksize
        self.last_chunk_len = self.h5len % chunksize
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.batch_size = batch_size
        self.chunk_batch_size = chunk_batch_size
        self.seqlen = seqlen
        self.n_features = n_features
        self.shuffle = shuffle
        self.rc = rc
        self.y_type = y_type
        self.len = np.min([len(self.chunks)*chunksize // self.batch_size, self.h5len // self.batch_size])
        self.on_epoch_end()

    def on_epoch_end(self):
        index_c = np.arange(len(self.chunks))
        if self.shuffle:
            np.random.shuffle(index_c)
        index_c = np.concatenate([ np.arange(c*self.chunksize, (c+1)*self.chunksize) if c != self.last_chunk else np.arange(c*self.chunksize,self.last_chunk_len) for c in index_c ])
        if self.shuffle:
            index_c = np.array_split(index_c, len(self.chunks) // self.chunk_batch_size )
            for i in range(len(index_c)):
                np.random.shuffle(index_c[i])
            index_c = np.concatenate(index_c)
        self.index_c = index_c

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # one index / batch
        indexes = self.index_c[index*self.batch_size:(index+1)*self.batch_size]
        indexes = np.sort(indexes)
        indexes = list(indexes)
        X, Y = self.__data_generation(indexes)
        return X, Y

    def __data_generation(self,indexes):
        rc = 0
        if self.rc:
            rc = choice([0,1])
        with h5py.File(self.h5file,'r') as f:
            data_x = f[self.x_dataset]
            data_y = f[self.y_dataset]
            X = np.empty((self.batch_size,self.seqlen,1,4),dtype='float32')
            Y = None
            if self.y_type == 1:
                Y = np.empty((self.batch_size, self.n_features), dtype='float32')
            elif self.y_type == 2:
                Y = np.empty((self.batch_size, self.n_features * 2), dtype='float32')
            x=data_x[indexes,]
            x=x[np.newaxis,]
            x=x.transpose((1,2,0,3))
            if rc:
                x=reverse_complement(x)
            X[:,] = x
            if self.y_type == 1:
                Y[:,] = data_y[indexes,]
            elif self.y_type == 2:
                y = data_y[indexes]
                Y[:,] = np.array([np.concatenate([(obs>1.).astype('float32'),obs]) for obs in y])
            else:
                raise NotImplementedError("type {} not implemented.".format(self.y_type))
            return X, Y

def MSLE_means_like(Y, sample=100000):
    s = np.min([len([Y]),sample])
    i = np.random.choice(len(Y),s)
    Y = Y[i]
    m = np.mean(Y,0)
    def MSLE_means(y_true, y_pred):
        first_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
        y_pred = K.zeros_like(y_true)
        y_new = y_pred + m
        second_log = K.log(K.clip(y_new, K.epsilon(), None) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1)
    return MSLE_means

def MSLE_ones(y_true, y_pred):
    first_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    second_log = K.log(K.ones_like(y_true) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)

