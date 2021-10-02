#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:02:17 2020

@author: moisey
"""
# Import necessary libraries for functions 
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Read in .csv file
dataSet = pd.read_csv("fitting_N20.csv")

# Extract the the x and y variables and use them for scatter plotting as Necessary
x = dataSet.iloc[:, 1]
t = dataSet.iloc[:, 2]
plt.scatter(x, t)

M = 5

# ================= Lets define the A matrix ====================
# Declare vars
temp = 0
rows, cols = (M, M) 
A = [[0]*cols]*rows
 
# get A matrix
for i in range(M):
    for j in range(M):
        for n in range(len(x)):
            temp += (x[n] ** (i+j)) 
        A[i][j] = temp
                     
# ================= Now define the T vector =====================
# Declare vars
temp = 0
rowVector = [0]*M

# get T vals
for i in range(M):
    for n in range(len(x)):
        temp += (x[n]**i) * t[n]
    rowVector[i] = temp
    
T = np.transpose(rowVector)

# Get w vector
Ainverse = np.linalg.pinv(A)

w = np.dot(Ainverse, T)


# ======================== plotting y(x,w) =======================

def y(x1,w):
    ans = 0
    for i in range(M):
        ans += (w[i] * x1**i)
    return ans

xInt = np.arange(0.0, 2.0, 0.01)
yInt = [0] * len(xInt)

for i in range(len(yInt)):
    yInt[i] = y(xInt[i], w)
    
    
# yy = w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3 + w[4]*x**4

# For some reason my function is not working as expected, despite
# trying bunch of things(for way too long) to plot and write the 
# function properly. Given the poly is of degree 4, I was aimming 
# for it to fit the 'W' shape of the data. Thanks for your consideration!

plt.plot(xInt,yInt)
plt.show()
