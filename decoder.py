# 2021.03.20
# @yifan
#
import numpy as np

from jpeg_utli import Shrink, invShrink, DCT, ZigZag
from jpeg_huffman import JPEG_Huffman_Luma, JPEG_Huffman_Chroma
from jpeg_header import JPEG_Header
            
class dJPEG_oneChannel(JPEG_Header, JPEG_Huffman_Luma, JPEG_Huffman_Chroma):
    def __init__(self, H, W, Qf, N, isluma=True, verbose=0):
        JPEG_Header.__init__(self, H=H, W=W, Qf=Qf, N=N)
        if isluma == True:
            JPEG_Huffman_Luma.__init__(self)
        else:
            JPEG_Huffman_Chroma.__init__(self)
        self.stream = ""
        self.ct = 0
        self.verbose = verbose

    def iDPCM(self, df):
        DC = np.zeros_like(df)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                DC[i,j,0] = df[i,j,0]
                for k in range(1, df.shape[2]):
                    DC[i,j,k] = DC[i,j,k-1] + df[i,j,k]
        return DC
    
    def decode_one_block(self):
        bX = np.zeros((self.N * self.N))
        for i in range(1, 30):
            if self.stream[self.ct:self.ct+i] in self.inv_DC_code_word:
                len_DC = self.inv_DC_code_word[self.stream[self.ct:self.ct+i]]
                self.ct += i
                if len_DC > 0:
                    bX[0] = self.bit2int(self.stream[self.ct:self.ct+(int)(len_DC)])
                    if self.verbose == 1:
                        print(self.stream[self.ct:self.ct+(int)(len_DC)], (int)(bX[0]))
                    self.ct += len_DC
                break
        n = 1
        while n < self.N * self.N:
            for i in range(1, 30):
                if self.stream[self.ct:self.ct+i] in self.inv_AC_code_word:
                    idx = self.inv_AC_code_word[self.stream[self.ct:self.ct+i]]
                    if self.verbose == 1:
                        print(idx, ', ', self.stream[self.ct:self.ct+i],end=', ')
                    self.ct += i
                    if idx == '0/0':
                        return bX
                    idx = idx.split('/')
                    n += (int)(idx[0])
                    if (int)(idx[1]) > 0:
                        bX[n] = self.bit2int(self.stream[self.ct:self.ct+(int)(idx[1])])
                        if self.verbose == 1:
                            print( (int)(bX[n]), self.stream[self.ct:self.ct+(int)(idx[1])])
                        self.ct += (int)(idx[1])
                        n += 1
                    break
        if n == self.N**2:
            if self.verbose == 1:
                print(stream[self.ct:self.ct+4])
            self.ct += 4
        return bX
    
    def decode(self, stream):
        self.stream = stream
        res = []
        num_block = 0
        while num_block < (self.H//self.N) * (self.W//self.N):
            res.append(self.decode_one_block())
            num_block += 1
            if self.verbose == 1:
                print(res[-1])
        res = np.array(res).reshape(1, self.H//self.N, self.W//self.N, -1)
        res[:,:,:,0] = self.iDPCM(res[:,:,:,0])
        res = ZigZag().inverse_transform(res)        
        res *= self.Qmatrix
        res = DCT(self.N, self.N).inverse_transform(res)
        iX = invShrink(res, self.N)
        return np.round(iX).astype('int16')

class dJPEG():
    def __init__(self, H, W, Qf, N, grayscale=False):
        self.H = H
        self.W = W
        self.Qf = Qf
        assert (N > 0), "<Error> Block size must greater than 0!"
        self.N = N
        self.grayscale = grayscale

    def write(self, X):
        if self.grayscale == False:
            X = c420_to_c444(X)
            X = BGR2YUV(X)
        else:
            X = X[0]
        return X

    def decode(self, stream):
        dJ = dJPEG_oneChannel(self.H, self.W, self.Qf, self.N, isluma=True, verbose=0)
        iX = [None] * 3
        iX[0] = dJ.decode(stream)[0,:,:,0]
        ct = dJ.ct
        if self.grayscale == False:
            for i in range(1, 3):
                dJ = dJPEG_oneChannel(self.H//2, self.W//2, self.Qf, self.N, isluma=False, verbose=0)
                iX[i] = dJ.decode(stream[ct:])[0,:,:,0]
        iX = self.write(iX)
        return iX