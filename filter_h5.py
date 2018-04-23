#!/usr/bin/env python

import argparse
import numpy as np
import h5py

def getargs():
    def ind(s):
        s = [ int(i) for i in s.split(',') ]
        return s
    parser = argparse.ArgumentParser()
    parser.add_argument('-onehot',required=True)
    parser.add_argument('-y',required=True)
    parser.add_argument('-sidoh',required=True)
    parser.add_argument('-sidy',required=True)
    parser.add_argument('-didoh',required=True)
    parser.add_argument('-didy',required=True)
    parser.add_argument('-o',required=True)
    parser.add_argument('-cs',required=False, type=int, default=100)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-keep',type=ind)
    g.add_argument('-drop',type=ind)
    parser.add_argument('-NN',required=False,default=10,type=int)
    parser.add_argument('-N0',required=False,default=1,type=int)
    parser.add_argument('-minE',required=False, default=1.,type=float)
    parser.add_argument('-d0',required=False,default=1,type=int)
    args = parser.parse_args()
    return args

class H5Parser:
    """
    loops over two h5 files, one containing the sequences (x) and the other containing the features (y)
    """
    def __init__(self,fx,fy,dx,dy,sx,sy,keep=None,drop=None,d0=1):
        self.fx = h5py.File(fx,'r')
        self.fy = h5py.File(fy,'r')
        self.x = self.fx[dx]
        self.y = self.fy[dy]
        self.sx = self.fx[sx]
        self.sy = self.fy[sy]
        self.d0 = d0
        d1 = int(d0!=1) 
        # print(self.y.shape)
        i = np.arange(self.y.shape[d1])
        if not drop is None:
            i = np.delete(i, drop)
        elif not keep is None:
            i = keep
        self.i = i
        self.chunks = self.x.chunks[0] # we will process one chunk at a time
        self.l = self.x.len()
        self.n_chunk = self.l // self.chunks
        assert self.l == self.y.shape[d0]
        
    def parse(self):
        x_batch = None
        y_batch = None
        r = 0 # current record
        n = 1
        while r < self.l:
            r_new = np.min([r+self.chunks,self.l])
            onehot_seq = self.sx[r:r_new]
            y_seq = self.sy[r:r_new]
            assert all(onehot_seq == y_seq)
            if self.d0 == 1:
                y_batch = self.y[self.i,r:r_new]
                y_batch = y_batch.transpose((1,0))
            else:
                y_batch = y[r:r_new,self.i]
            x_batch = self.x[r:r_new,:,:]
            print('processed {} out of {} chunks.'.format(n, self.n_chunk))
            yield(x_batch, y_batch, onehot_seq)
            n += 1
            r = r_new            
            
    def close(self):
        self.fx.close()
        self.fy.close()
        
def main():
    args = getargs()
    def filter_x(x, NN=args.NN):
        n_N = np.sum(x,1)[:,3]
        return n_N < NN
    def filter_y(y, N0=args.N0, minE=args.minE):
        n_0 = np.sum( y > 1., 1)
        maxE = np.max(y, 1)
        return (n_0 >= N0) & (maxE >= minE) 
    p = H5Parser(args.onehot, args.y, args.didoh, args.didy, args.sidoh, args.sidy, keep=args.keep, drop=args.drop, d0=args.d0) 
    cs = args.cs
    
    with h5py.File(args.o, 'w') as outfile:
        outfile.create_dataset('seqID', shape=(1,), maxshape=(None,), chunks=(cs,),dtype=p.sx.dtype)
        outfile.create_dataset('onehot', shape=(1,1000,4), maxshape=(None,1000,4), chunks=(cs,1000,4),dtype='i2')
        outfile.create_dataset('y', shape=(1,len(p.i)),maxshape=(None,len(p.i)), chunks=(cs,len(p.i)), dtype=p.y.dtype)
        r = 0
        r_new = 0
        for x_batch, y_batch, seq_batch in p.parse():
            keep = filter_x(x_batch) & filter_y(y_batch)
            x_batch = x_batch[keep]
            x_batch = np.delete(x_batch,3,2)
            y_batch = y_batch[keep]
            seq_batch=seq_batch[keep]
            r_new += np.sum(keep)
            if r_new > r:
                outfile['seqID'].resize((r_new), axis = 0 )
                outfile['onehot'].resize((r_new), axis = 0 )
                outfile['y'].resize((r_new), axis = 0)
                outfile['seqID'][r:r_new] = seq_batch
                outfile['onehot'][r:r_new] = x_batch
                outfile['y'][r:r_new] = y_batch
            r = r_new
        p.close()

if __name__=="__main__":
    main()
