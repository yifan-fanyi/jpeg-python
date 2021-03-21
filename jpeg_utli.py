# 2021.03.20
# @yifan
#
import numpy as np
from skimage.util import view_as_windows
from scipy.fftpack import dct, idct

def Shrink(X, win):
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, win):
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

class DCT():
    def __init__(self, N=8, P=8):
        self.N = N
        self.P = P
        self.W = 8
        self.H = 8
    
    def transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.N, self.P, 1)
        a = dct(dct(a, axis=1, norm='ortho'), axis=2, norm='ortho')
        return a.reshape(S)

    def inverse_transform(self, a):
        S = list(a.shape)
        a = a.reshape(-1, self.N, self.P, 1)
        a = idct(idct(a, axis=1, norm='ortho'), axis=2, norm='ortho')
        return a.reshape(S)
    
    def ML_inverse_transform(self, Xraw, X):
        llsr = LLSR(onehot=False)
        llsr.fit(X.reshape(-1, X.shape[-1]), Xraw.reshape(-1, X.shape[-1]))
        S = X.shape
        X = llsr.predict_proba(X.reshape(-1, X.shape[-1])).reshape(S)
        return X

class ZigZag():
    def __init__(self):
        self.idx = []
        
    def zig_zag(self, i, j, n):
        if i + j >= n:
            return n * n - 1 - self.zig_zag(n - 1 - i, n - 1 - j, n)
        k = (i + j) * (i + j + 1) // 2
        return k + i if (i + j) & 1 else k + j

    def zig_zag_getIdx(self, N):
        idx = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                idx[i, j] = self.zig_zag(i, j, N)
        return idx.reshape(-1)
    
    def transform(self, X):
        self.idx = self.zig_zag_getIdx((int)(np.sqrt(X.shape[-1]))).astype('int32')
        S = list(X.shape)
        X = X.reshape(-1, X.shape[-1])
        return X[:, np.argsort(self.idx)].reshape(S)
    
    def inverse_transform(self, X):
        self.idx = self.zig_zag_getIdx((int)(np.sqrt(X.shape[-1]))).astype('int32')
        S = list(X.shape)
        X = X.reshape(-1, X.shape[-1])
        return X[:, self.idx].reshape(S)
        
class LLSR():
    def __init__(self, onehot=True, normalize=False):
        self.onehot = onehot
        self.normalize = normalize
        self.weight = []

    def fit(self, X, Y):
        if self.onehot == True:
            Y = np.eye(len(np.unique(Y)))[Y.reshape(-1)]
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        self.weight, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def predict(self, X):
        pred = self.predict_proba(X)
        return np.argmax(pred, axis=1)

    def predict_proba(self, X):
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        pred = np.matmul(X, self.weight)
        if self.normalize == True:
            pred = (pred - np.min(pred, axis=1, keepdims=True))/ np.sum((pred - np.min(pred, axis=1, keepdims=True) + 1e-15), axis=1, keepdims=True)
        return pred

    def score(self, X, Y):
        pred = self.predict(X)
        return accuracy_score(Y, pred)