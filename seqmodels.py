
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
    print("Conv1a shape: {}".format(conv1.shape))
    conv2 = Conv2D(n_f1, (3, 1) , activation="relu", padding="same")(conv1)
    print("Conv1b shape: {}".format(conv2.shape))
    pool1 = MaxPooling2D(pool_size=(2, 1),padding='valid')(conv2)
    print("Pool1 shape: {}".format(pool1.shape))
    conv3 = Conv2D(256, (3,1), activation="relu", padding="same")(pool1)
    print("Conv2a shape: {}".format(conv3.shape))
    conv3b = Conv2D(256, (3,1), activation="relu", padding="same")(conv3)
    print("Conv2b shape: {}".format(conv3b.shape))
    pool2 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv3)
    print("Pool2 shape: {}".format(pool2.shape))
    conv4 = Conv2D(512, (3,1), activation="relu", padding="same")(pool2)
    print("Conv3 shape: {}".format(conv4.shape))
    pool3 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv4)
    print("Pool3 shape: {}".format(pool3.shape))
    conv5 = Conv2D(1024, (3,1), activation="relu", padding="same")(pool3)
    print("Conv4 shape: {}".format(conv5.shape))
    pool4 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv5)
    print("Pool4 shape: {}".format(pool4.shape))
    # regularizer is applied to the last layers, if needed
    r = regularizers.l2(reg)
    flat = Flatten()(pool4)
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

def pwm_simple_v2(seqlen=1000, n_features=1000, reg=0, n_f1=128, len_f1=3, data_format="channels_last", mtype=1):
    x = Input(batch_shape=(None,seqlen,1,4))
    print("Input shape: {0}".format(x.shape))
    f = (len_f1, 1)
    conv1 = Conv2D(n_f1, f, activation="relu", padding="same")(x)
    print("Conv1a shape: {}".format(conv1.shape))
    conv2 = Conv2D(n_f1, (3, 1) , activation="relu", padding="same")(conv1)
    print("Conv1b shape: {}".format(conv2.shape))
    pool1 = MaxPooling2D(pool_size=(2, 1),padding='valid')(conv2)
    print("Pool1 shape: {}".format(pool1.shape))
    conv3 = Conv2D(256, (3,1), activation="relu", padding="same")(pool1)
    print("Conv2a shape: {}".format(conv3.shape))
    conv3b = Conv2D(256, (3,1), activation="relu", padding="same")(conv3)
    print("Conv2b shape: {}".format(conv3b.shape))
    pool2 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv3)
    print("Pool2 shape: {}".format(pool2.shape))
    conv4 = Conv2D(512, (3,1), activation="relu", padding="same")(pool2)
    print("Conv3 shape: {}".format(conv4.shape))
    pool3 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv4)
    print("Pool3 shape: {}".format(pool3.shape))
    conv5 = Conv2D(1024, (3,1), activation="relu", padding="same")(pool3)
    print("Conv4 shape: {}".format(conv5.shape))
    pool4 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv5)
    print("Pool4 shape: {}".format(pool4.shape))
    # regularizer is applied to the last layers, if needed
    r = regularizers.l2(reg)
    flat = Flatten()(pool4)
    print("Flat shape: {}".format(flat.shape))
    dense1 = Dense(128, activation = "linear", kernel_regularizer = r, bias_regularizer = r, name="bottleneck")(flat)
    dense1 = Activation("relu")(dense1)
    print("Dense1 shape: {}".format(dense1.shape))
    dense2 = Dense(256, activation = "relu", kernel_regularizer = r, bias_regularizer = r)(dense1)
    print("Dense2 shape: {}".format(dense2.shape))
    dense3 = Dense(512, activation = "relu", kernel_regularizer = r, bias_regularizer = r)(dense2)
    print("Dense3 shape: {}".format(dense3.shape))
    dense4 = Dense(n_features, activation="linear", kernel_regularizer=r, bias_regularizer=r)(dense3)
    print("Dense4 shape: {}".format(dense4.shape))
    if mtype == 1:
        model = Model(inputs=x, outputs=dense2)
        print("Output shape: {}".format(dense4.shape))
        return model
    elif mtype == 2:
        dense4b = Dense(n_features, activation="sigmoid", kernel_regularizer=r, bias_regularizer=r)(dense3)
        print("Dense4b shape: {}".format(dense4b.shape))
        out = Concatenate(axis=1)([dense4b, dense4])
        print("Output shape: {}".format(out.shape))
        model = Model(inputs=x, outputs=out)
        return model
    else:
        raise NotImplementedError("mtype {} not implemented.".format(mtype))

def pwm_simple_v3(seqlen=1000, n_features=1000, reg=0, n_f1=128, len_f1=3, data_format="channels_last", mtype=1):
    x = Input(batch_shape=(None,seqlen,1,4))
    print("Input shape: {0}".format(x.shape))
    f = (len_f1, 1)
    conv1 = Conv2D(n_f1, f, activation="relu", padding="same")(x)
    print("Conv1a shape: {}".format(conv1.shape))
    conv2 = Conv2D(n_f1, (3, 1) , activation="relu", padding="same")(conv1)
    print("Conv1b shape: {}".format(conv2.shape))
    pool1 = MaxPooling2D(pool_size=(2, 1),padding='valid')(conv2)
    print("Pool1 shape: {}".format(pool1.shape))
    conv3 = Conv2D(256, (3,1), activation="relu", padding="same")(pool1)
    print("Conv2a shape: {}".format(conv3.shape))
    conv3b = Conv2D(256, (3,1), activation="relu", padding="same")(conv3)
    print("Conv2b shape: {}".format(conv3b.shape))
    pool2 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv3)
    print("Pool2 shape: {}".format(pool2.shape))
    conv4 = Conv2D(512, (3,1), activation="relu", padding="same")(pool2)
    print("Conv3 shape: {}".format(conv4.shape))
    pool3 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv4)
    print("Pool3 shape: {}".format(pool3.shape))
    conv5 = Conv2D(1024, (3,1), activation="relu", padding="same")(pool3)
    print("Conv4 shape: {}".format(conv5.shape))
    pool4 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv5)
    print("Pool4 shape: {}".format(pool4.shape))
    conv6 = Conv2D(2028, (3,1), activation="relu", padding="same")(pool4)
    print("Conv5 shape: {}".format(conv6.shape))
    pool5 = MaxPooling2D(pool_size=(2, 1), padding="valid")(conv6)
    print("Pool5 shape: {}".format(pool5.shape))
    # regularizer is applied to the last layers, if needed
    r = regularizers.l2(reg)
    flat = Flatten()(pool5)
    print("Flat shape: {}".format(flat.shape))
    dense1 = Dense(128, activation = "linear", kernel_regularizer = r, bias_regularizer = r, name="bottleneck")(flat)
    dense1 = Activation("relu")(dense1)
    print("Dense1 shape: {}".format(dense1.shape))
    dense2 = Dense(256, activation = "relu", kernel_regularizer = r, bias_regularizer = r)(dense1)
    print("Dense2 shape: {}".format(dense2.shape))
    dense3 = Dense(512, activation = "relu", kernel_regularizer = r, bias_regularizer = r)(dense2)
    print("Dense3 shape: {}".format(dense3.shape))
    dense4 = Dense(n_features, activation="linear", kernel_regularizer=r, bias_regularizer=r)(dense3)
    print("Dense4 shape: {}".format(dense4.shape))
    if mtype == 1:
        model = Model(inputs=x, outputs=dense2)
        print("Output shape: {}".format(dense4.shape))
        return model
    elif mtype == 2:
        dense4b = Dense(n_features, activation="sigmoid", kernel_regularizer=r, bias_regularizer=r)(dense1)
        print("Dense4b shape: {}".format(dense2.shape))
        out = Concatenate(axis=1)([dense4b, dense4])
        print("Output shape: {}".format(out.shape))
        model = Model(inputs=x, outputs=out)
        return model
    else:
        raise NotImplementedError("mtype {} not implemented.".format(mtype))

def reverse_complement(onehotseq):
    return onehotseq[:,::-1,:,::-1]

def weighted_seqloss(w,l='mse'):
    weight = w
    def lossfun(y_true, y_pred):
        y_pred_prob, y_pred_value = tf.split(y_pred,2,axis=-1)
        y_true_prob, y_true_value = tf.split(y_true,2,axis=-1)
        crossentrop = losses.binary_crossentropy(y_true_prob, y_pred_prob)
        zero = K.constant([0.], dtype='float32')
        zeros = tf.zeros_like(y_true_value)
        eps = K.epsilon()      
        y_true_value_filter=tf.where(tf.cast(y_true_prob, bool), y_true_value, zeros)
        y_pred_value_filter=tf.where(tf.cast(y_true_prob, bool), y_pred_value, zeros)
        n = K.sum(y_true_prob,axis=-1)
        # n = K.cast(n, 'float32')
        zeros_only =  K.equal(n, zero)
        err = None
        if l == 'mse':
            err = y_true_value_filter - y_pred_value_filter
            err = K.sum(K.square(err), axis=-1)
        elif l == 'msle':
            first_log = K.log(K.clip(y_true_value_filter, eps, None) + 1.)
            second_log = K.log(K.clip(y_pred_value_filter, eps, None) + 1.)
            err = K.sum(K.square(first_log - second_log), axis=-1)
        elif l == 'mae':
            err = y_true_value_filter - y_pred_value_filter
            err = K.sum(K.abs(err), axis=-1)
        loss = tf.where(zeros_only, (1-weight)*crossentrop, (1-weight)*crossentrop + weight*err/n)
        return loss
    return lossfun

def seqloss1(y_true, y_pred):
    w = 0.5
    y_pred_prob, y_pred_value = tf.split(y_pred,2,axis=-1)
    y_true_prob, y_true_value = tf.split(y_true,2,axis=-1)
    err = y_true_value - y_pred_value
    z = tf.zeros_like(err)
    err = tf.where(tf.cast(y_true_prob,bool),err,z)
    sqerr = tf.reduce_sum(tf.square(err))
    crossentrop = losses.binary_crossentropy(y_true_prob, y_pred_prob)
    return (1-w)*crossentrop + w*sqerr

def seq_avg_recall(y_true, y_pred):
    one = tf.constant(1, dtype=tf.float32)
    pointfive = tf.constant(0.5, dtype=tf.float32)
    y_pred_prob, _ = tf.split(y_pred,2,axis=-1)
    y_true_prob, _ = tf.split(y_true,2,axis=-1)
    zeros = tf.zeros_like(y_true_prob)
    n_true = tf.reduce_sum(y_true_prob)
    p_pos = tf.where(tf.equal(y_true_prob,one), y_pred_prob, zeros)
    n_pos = tf.reduce_sum(tf.cast(tf.greater_equal(p_pos, pointfive), tf.float32))
    return n_pos / n_true

def seq_avg_precision(y_true, y_pred):
    one = tf.constant(1, dtype=tf.float32)
    pointfive = tf.constant(0.5, dtype=tf.float32)
    y_pred_prob, _ = tf.split(y_pred,2,axis=-1)
    y_true_prob, _ = tf.split(y_true,2,axis=-1)
    zeros = tf.zeros_like(y_true_prob)
    n_pred_true = tf.reduce_sum(tf.cast(tf.greater_equal(y_pred_prob, pointfive), tf.float32))
    p_pos = tf.where(tf.equal(y_true_prob,one), y_pred_prob, zeros)
    n_pos = tf.reduce_sum(tf.cast(tf.greater_equal(p_pos, pointfive), tf.float32))
    return n_pos / (n_pred_true+1)

def seq_avg_sqerr(y_true, y_pred):
    y_pred_prob, y_pred_value = tf.split(y_pred,2,axis=-1)
    y_true_prob, y_true_value = tf.split(y_true,2,axis=-1)
    n = K.sum(y_true_prob,axis=-1)
    zero = K.zeros_like(n, dtype='float32')
    all_zero = K.equal(n, zero)
    zeros = K.zeros_like(y_true_value)
    err = y_pred_value - y_true_value
    err = tf.where(tf.cast(y_true_prob, bool), err, zeros)
    err = K.sum(K.square(err), axis=-1)
    r = tf.where(all_zero, zero, err/n)
    return r

def seq_avg_sqlogerr(y_true, y_pred):
    y_pred_prob, y_pred_value = tf.split(y_pred,2,axis=-1)
    y_true_prob, y_true_value = tf.split(y_true,2,axis=-1)
    n = K.sum(y_true_prob, axis=-1)
    zero = K.zeros_like(n)
    all_zero=K.equal(n, zero)
    zeros = tf.zeros_like(y_true_value)
    first_log = K.log(K.clip(y_pred_value, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true_value, K.epsilon(), None) + 1.)
    err = first_log - second_log
    err = tf.where(tf.cast(y_true_prob, bool), err, zeros)
    err = K.sum(K.square(err), axis=-1)
    r = tf.where(all_zero, zero, err/n)
    return r

def seq_avg_sqerr_1(y_true, y_pred):
    y_pred_value = y_pred
    y_true_value = y_true
    ones = K.ones_like(y_true,dtype='float32')
    y_true_prob = K.cast(K.not_equal(y_true_value, ones),'float32')    
    n = K.sum(y_true_prob,axis=-1)
    zero = K.zeros_like(n, dtype='float32')
    all_zero = K.equal(n, zero)
    zeros = K.zeros_like(y_true_value)
    err = y_pred_value - y_true_value
    err = tf.where(tf.cast(y_true_prob, bool), err, zeros)
    err = K.sum(K.square(err), axis=-1)
    r = tf.where(all_zero, zero, err/n)
    return r

def seq_avg_sqerr_1_a0(y_true, y_pred):
    y_true_value = y_true
    ones = K.ones_like(y_true,dtype='float32')
    y_pred_value = ones
    y_true_prob = K.cast(K.not_equal(y_true_value, ones),'float32')
    n = K.sum(y_true_prob,axis=-1)
    zero = K.zeros_like(n, dtype='float32')
    all_zero = K.equal(n, zero)
    zeros = K.zeros_like(y_true_value)
    err = y_pred_value - y_true_value
    err = tf.where(tf.cast(y_true_prob, bool), err, zeros)
    err = K.sum(K.square(err), axis=-1)
    r = tf.where(all_zero, zero, err/n)
    return r

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

