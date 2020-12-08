# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 17:07:32 2020
@author: Mingfen Wang
"""

import numpy as np

def coordinate_descent_method(A, b, guess, N, TOR):
    A_T = A.T
    x = guess.copy()
    k = 1
    while k < N:
        save_x = x.copy()
        for i in range(len(x)):
            b_in_Ax = b - np.dot(A, x)
            foot = np.dot(b_in_Ax, A_T[i]) / np.dot(A_T[i], A_T[i])
            x[i] = x[i] + foot
        diff_x = x - save_x
        if np.dot(diff_x, diff_x) < TOR:
            break
        k = k + 1
    return x, k

A = np.array([[1., -1., 0.],\
              [-1., 2., 1.],\
              [0., 1., 5.]])
b = np.array([3., -3., 4.])
guess = np.array([0., 0., 0.])
x, k = coordinate_descent_method(A, b, guess, 300 * len(b), 1e-6)
print("x = {} in {} iterations.".format(x.round(2), k))
