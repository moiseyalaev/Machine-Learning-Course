#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:44:06 2020

@author: moisey
"""

# Import necessary libraries for functions 
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random

# Read in .csv file
dataSet = pd.read_csv("winequality-red.csv", delimiter=";")

# Define the design matrix (matrix of the feature maps) 
X = np.asarray(dataSet.iloc[:,range(11)])

# Define target values
t = dataSet.loc[ :,"quality"]

XT = X.T # X transpose
N = len(X)

# ====================== Computing wStar directly using psuedoinverse ===================

# Compute wStar
pseudoinverse = np.dot(np.linalg.inv(np.dot(XT, X)), XT)
wStar = np.dot(pseudoinverse, t)

print(wStar[range(5)])
# Output: [ 0.00419374 -1.0997431  -0.18414597  0.00707117 -1.91141882]

# Compute average error
error1 = np.linalg.norm(np.subtract(np.dot(X,wStar), t)) ** 2 / N
print(error1)
# outout: 0.41704922482048457
# This error is very low as it should define the least theoretic error since w is derived
# from maximizing the likelihood functon.

# =========================================== LMS ======================================

# Make a function that handles SGD, produces an error (from wStar), 
# plots this error at each iteratio, and returns optimal weights
def LMS(t, X, wStar, iterations):
    w = np.asarray([0] * 11)
    error = [0] * iterations
    
    for k in range(iterations):
       
            randNum = random.randint(0, N-1)
            Xn = X[randNum][:]
            
            step = np.linalg.norm(Xn)**-2
      
            inParen = np.subtract(t[randNum], np.dot(w.T, Xn))
            arrProd = np.dot(inParen, Xn)
    
            w = np.add(w, step * arrProd)
            
            error[k] = np.linalg.norm(wStar - w)
            
    plt.plot(range(iterations), error)

    return w

        
w = LMS(t, X, wStar, 1000)

error2 = np.linalg.norm(np.subtract(np.dot(X, w), t)) ** 2 / N
print(error2)
# Error fluctuates depending on random values extracted from X and t




        