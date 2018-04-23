#!/usr/bin/env python
import seqmodels
from seqmodels import seq_avg_recall, seq_avg_precision, seq_avg_sqerr, seq_avg_sqlogerr, seq_avg_sqerr_1, seq_avg_sqerr_1_a0
import h5py
import numpy as np
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
import argparse
# import importlib

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('h5')
    p.add_argument('-mtype', default=2, type=int)
    p.add_argument('-w', help='weight of loss terms (1-w)*binary_crossentropy + w*l', default=0.01,type=float)
    p.add_argument('-l',choices=['mse','msle','msae'],help='mse=mean_squared_error, msle=mean_squared_logarithmic_error, msae=mean_absolute_error', default='msle')
    p.add_argument('-lr',type=float,default=0.0001)
    p.add_argument('-load_weights')
    p.add_argument('-n_epoch',default=1,type=int)
    p.add_argument('-prefix',default="")
    p.add_argument('-rc', type=bool, default=False)
    args = p.parse_args()
    return args

def main():
    args = get_args()
    f, y_type, w, l, lr, weights, n_epoch, prefix, rc= args.h5, args.mtype, args.w, args.l, args.lr, args.load_weights, args.n_epoch, args.prefix, args.rc

    model = seqmodels.pwm_simple(n_features=1986,len_f1=3,mtype=y_type)
    if not weights is None:
        model.load_weights(weights)

    train_i = np.arange(0,16000)
    valid_i = np.arange(16000,19137)

    if prefix != "":
        if not prefix.endswith("_"):
            prefix += "_"

    loss = None
    metrics = None
    if y_type == 2:
        loss = seqmodels.weighted_seqloss(w, l)
        metrics = [seq_avg_recall, seq_avg_precision, seq_avg_sqerr, seq_avg_sqlogerr]
    elif y_type == 1:
        if l == 'mse':
            loss = keras.losses.mean_squared_error
        elif l == 'msle':
            loss = keras.losses.mean_squared_logarithmic_error
        elif l == 'msae':
            loss = keras.losses.mean_absolute_error
        metrics = [seq_avg_sqerr_1, seq_avg_sqerr_1_a0]

    train_gen = seqmodels.DataGenerator(train_i, 100, f, 'onehot', 'y', n_features=1986, chunk_batch_size=10, y_type=y_type, shuffle=True, rc=rc)
    valid_gen = seqmodels.DataGenerator(valid_i, 100, f, 'onehot', 'y', n_features=1986, chunk_batch_size=10, y_type=y_type, shuffle=False, rc=False)

    model_checkpoint = ModelCheckpoint('{}model.hdf5'.format(prefix), monitor='val_loss',verbose=1, save_best_only=True)
    csv_logger = CSVLogger('{}training.log'.format(prefix),append=True)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    model.fit_generator(generator=train_gen, validation_data=valid_gen, epochs=n_epoch, verbose=1, callbacks=[model_checkpoint,csv_logger], use_multiprocessing=False)

if __name__ == "__main__":
    main()
