#!/usr/bin/env python

from pyfaidx import Fasta
from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import argparse
import h5py

# preprocess fasta with "reformat.sh in=seqs.fa out=out.fa iupacton"

class FastaOnehot:
    def __init__(self,file=None,seqlen=1000):
        self.fa = Fasta(file,sequence_always_upper=True,as_raw=True)
        self.l = len(self.fa.keys())
        self.seqlen = seqlen
        self.dna_encoder=LabelEncoder().fit(array(['A','C','G','N','T']))
        self.onehot_encoder = OneHotEncoder(sparse=False).fit(array(list(range(0,5))).reshape(-1, 1))

    def toOnehot(self,chunksize=100000):
        k = list(self.fa.keys())
        r = 0
        i = 0
        # TODO: make this accept variable length sequences -> pad or crop if length =/= seqlen
        while r < self.l:
            seqnames = k[ i*chunksize:min((i+1)*chunksize,self.l)]
            seq = [ array(list(self.fa[x][:].ljust(1000,'N'))) for x in seqnames ]
            int_encoded = [ self.dna_encoder.transform(s) for s in seq ]
            int_encoded = [ s.reshape(len(s),1) for s in int_encoded ]
            onehot_encoded = array([self.onehot_encoder.transform(s) for s in int_encoded ])
            r = min((i+1)*chunksize,self.l)
            i += 1
            print('last record : '+seqnames[-1])
            yield (seqnames,onehot_encoded)

    def toh5(self, file="onehot.h5", chunksize=100000):
        with h5py.File(file, 'w') as f:
            cs = min(chunksize, self.l)
            f.create_dataset("seqnames", data = [np.string_(s) for s in list(self.fa.keys())], chunks=(cs,))
            f.create_dataset("onehot",shape=(self.l,self.seqlen,5),maxshape=(None,self.seqlen,5), chunks=(cs,self.seqlen,5))
            i = 0
            for n, o in self.toOnehot(chunksize=chunksize):
                f["onehot"][(i*cs):min(((i+1)*cs),self.l)] = o
                i+=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help="Input FASTA file")
    parser.add_argument('h5', help = "Output HDF5 file")
    parser.add_argument('chunksize', help="Chunks for reading/writing fasta/onehot records")
    parser.add_argument('seqlen', help="DNA sequence length", default=1000)
    args=parser.parse_args()
    foh = FastaOnehot(args.fasta, seqlen=int(args.seqlen))
    foh.toh5(args.h5, int(args.chunksize))

if __name__ == "__main__":
    main()

# def fastaOnehot(fa,chunksize=100000):
#     fa = Fasta(fa,sequence_always_upper=True,as_raw=True)

#     k = list(fa.keys())
#     l = len(k)
#     r = 0
#     i = 0

#     dna_encoder=LabelEncoder().fit(array(['A','C','G','N','T']))
#     onehot_encoder = OneHotEncoder(sparse=False).fit(array(list(range(0,5))).reshape(-1, 1))

#     while r < l:
#         seqnames = k[ i*chunksize:min((i+1)*chunksize,l)]
#         seq = [ array(list(fa[x][:])) for x in seqnames ]
#         int_encoded = [ dna_encoder.transform(s) for s in seq ]
#         int_encoded = [ s.reshape(len(s),1) for s in int_encoded ]
#         onehot_encoded = array([onehot_encoder.transform(s) for s in int_encoded ])
        
#         r = min((i+1)*chunksize,l)
#         i += 1
#         print('last record : '+seqnames[-1])
#         yield (seqnames,c,onehot_encoded)

