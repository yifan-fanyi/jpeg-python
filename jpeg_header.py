# 2021.03.20
# @yifan
#
import numpy as np
import cv2

class JPEG_Header():
    def __init__(self, H, W, Qf, N):
        self.H = H
        self.W = W
        self.Qf = Qf
        self.N = N
        self.JPEG_Qmatrix = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                      12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77,
                      24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101,
                      72, 92, 95, 98, 112, 100, 103, 99], dtype='float64')
        if self.Qf >= 50:   
            self.Qmatrix = cv2.resize((100. - self.Qf) / 50. * self.JPEG_Qmatrix.reshape(8,8), (self.N, self.N)).reshape(-1)
        else:
            self.Qmatrix = cv2.resize(50. / self.Qf * self.JPEG_Qmatrix.reshape(8,8), (self.N, self.N)).reshape(-1)

