# 2021.03.20
# @yifan
#
import numpy as np

from jpeg_utli import Shrink, invShrink, DCT, ZigZag
from jpeg_huffman import JPEG_Huffman_Luma
from jpeg_header import JPEG_Header

class eJPEG_oneChannel(JPEG_Header, JPEG_Huffman_Luma):
    def __init__(self, H, W, Qf=50, N=8, verbose=0):
        JPEG_Header.__init__(self, H=H, W=W, Qf=Qf, N=N)
        JPEG_Huffman_Luma.__init__(self)
        self.stream = ""
        self.verbose = verbose
       
    def DPCM(self, DC):
        df = np.zeros_like(DC)
        for i in range(DC.shape[0]):
            for j in range(DC.shape[1]):
                df[i,j,0] = DC[i,j,0]
                for k in range(1, DC.shape[2]):
                    df[i,j,k] = DC[i,j,k] - DC[i,j,k-1]
        return df
    
    def DC_Huffman(self, DC):
        cat = self.Category(DC)
        if cat == 0:
            return self.DC_code_word[cat]
        return self.DC_code_word[cat] + self.int2bit(DC, cat)
    
    def AC_Huffman(self, AC):
        AC = AC.reshape(-1).astype('int16')
        ct = 0
        bits = ''
        for i in range(len(AC)):
            if AC[i] == 0:
                ct += 1
            else:
                while ct > 15:
                    bits += self.AC_code_word['15/0'] 
                    ct -= 15 
                cat = self.Category(AC[i])
                bits += self.AC_code_word[str(ct)+'/'+str(cat)] + self.int2bit(AC[i], cat)
                if self.verbose == 1:
                    print(str(ct)+'/'+str(cat), self.AC_code_word[str(ct)+'/'+str(cat)] , ',',self.int2bit(AC[i], cat))
                ct = 0
        bits += self.AC_code_word['0/0']
        return bits
    
    def encode_one_block(self, X):
        X = X.reshape(1, -1).astype('int16')
        X = ZigZag().transform(X)
        self.stream += self.DC_Huffman(X[0, 0])
        self.stream += self.AC_Huffman(X[0, 1:])
        if self.verbose == 1:
            print(X)
        
    def encode(self, X):
        X = Shrink(X, self.N)
        tX = DCT(self.N, self.N).transform(X)
        qX = np.round(tX/self.Qmatrix)
        qX[:,:,:,0] = self.DPCM(qX[:,:,:,0])
        qX = qX.reshape(-1, qX.shape[-1])
        for i in range(qX.shape[0]):
            self.encode_one_block(qX[i])
        return self.stream
            
class dJPEG_oneChannel(JPEG_Header, JPEG_Huffman_Luma):
    def __init__(self, H, W, Qf=50, N=8, verbose=0):
        JPEG_Header.__init__(self, H=H, W=W, Qf=Qf, N=N)
        JPEG_Huffman_Luma.__init__(self)
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
        while self.ct < len(self.stream):
            res.append(self.decode_one_block())
            if self.verbose == 1:
                print(res[-1])
        res = np.array(res).reshape(1, self.H//self.N, self.W//self.N, -1)
        res[:,:,:,0] = self.iDPCM(res[:,:,:,0])
        res = ZigZag().inverse_transform(res)        
        res *= self.Qmatrix
        res = DCT(self.N, self.N).inverse_transform(res)
        iX = invShrink(res, self.N)
        return np.round(iX).astype('int16')