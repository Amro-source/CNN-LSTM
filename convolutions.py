# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:13:05 2020

@author: Zikantika
"""


from scipy import signal
from scipy import misc
import numpy as np
from numpy import zeros
 
def unfold_matrix(X, k):
    n, m = X.shape[0:2]
    xx = zeros(((n - k + 1) * (m - k + 1), k**2))
    row_num = 0
    def make_row(x):
        return x.flatten()
 
    for i in range(n- k+ 1):
        for j in range(m - k + 1):
            #collect block of m*m elements and convert to row
            xx[row_num,:] = make_row(X[i:i+k, j:j+k])
            row_num = row_num + 1
 
    return xx
 
w = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]], np.float32)
#x = np.random.randn(5,5)
x = np.array([[-0.21556299, -0.11002319, -0.3499612,   1.49290769, -0.50435978],
 [ 0.06348409,  0.66873375,  0.14251138, -1.6414004 , -0.91561852],
 [-2.52451962, -1.97544675, -0.24609529, -1.11489934, -1.44793437],
 [ 1.26260575, -0.62047366,  0.12274525,  0.25200227, -0.83925847],
 [-1.54336488, -0.05100702,  0.36608208,  0.51712927, -0.97133877],
[-1.54336488, -0.05100702, 0.36608208, 0.51712927, -0.97133877]])
 
n, m = x.shape[0:2]
k = w.shape[0]
 
y = signal.correlate2d(x, w, mode='valid')
 
x_unfolded = unfold_matrix(x, k)
w_flat = w.flatten()
yy = np.matmul(x_unfolded, w_flat)
yy = yy.reshape((n-k+1, m-k+1))
print(yy)
# verify yy = y