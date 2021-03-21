# 2021.03.20
# @yifan
#
# JPEG encoder, the header part are not included
# only the DC and the AC bitstream included
import numpy as np
import cv2

from jpeg_utli import Shrink, invShrink, DCT, ZigZag
from jpeg_huffman import JPEG_Huffman_Luma, JPEG_Huffman_Chroma
from jpeg_header import JPEG_Header
from jpeg_color import BGR2YUV, c444_to_c420

class eJPEG_oneChannel(JPEG_Header, JPEG_Huffman_Luma, JPEG_Huffman_Chroma):
    def __init__(self, H, W, Qf, N, isluma=True, verbose=0):
        JPEG_Header.__init__(self, H=H, W=W, Qf=Qf, N=N)
        if isluma == True:
            JPEG_Huffman_Luma.__init__(self)
        else:
            JPEG_Huffman_Chroma.__init__(self)
        self.isluma = isluma
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
            
class eJPEG():
    def __init__(self, Qf, N, grayscale=False):
        self.Qf = Qf
        assert (N > 0), "<Error> Block size must greater than 0!"
        self.N = N
        self.grayscale = grayscale
        self.stream = ""

    def load(self, img):
        try:
            X = cv2.imread(img)
            X.shape
        except:
            assert (False), "<Error> Cannot load image: " + img
        X = BGR2YUV(X)
        X = c444_to_c420(X)
        return X

    def encode(self, img):
        X = self.load(img)
        H, W = X[0].shape[0], X[0].shape[1]
        eJ = eJPEG_oneChannel(H, W, self.Qf, self.N, isluma=True, verbose=0)
        self.stream += eJ.encode(X[0].reshape(1, H, W, 1))
        if self.grayscale == False:
            for i in range(1, 3):
                eJ = eJPEG_oneChannel(H//2, W//2, self.Qf, self.N, isluma=False, verbose=0)
                self.stream += eJ.encode(X[i].reshape(1, H//2, W//2, 1))
        return self.stream