# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:48:22 2020

@author: Zikantika
"""

# importing required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.datasets import mnist

# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# selecting a subset of data (200 images)
x_train = x_train[:200]
y = y_train[:200]

X = x_train.T
X = X/255

y.resize((200,1))
y = y.T

#checking value
pd.Series(y[0]).value_counts()

# converting into binary classification
for i in range(y.shape[1]):
    if y[0][i] >4:
        y[0][i] = 1
    else:
        y[0][i] = 0

#checking value counts
pd.Series(y[0]).value_counts()

# initializing filter
f=np.random.uniform(size=(3,5,5))
f = f.T

print('Filter 1', '\n', f[:,:,0], '\n')
print('Filter 2', '\n', f[:,:,1], '\n')
print('Filter 3', '\n', f[:,:,2], '\n')

# Generating patches from images
new_image = []

# for number of images
for k in range(X.shape[2]):
    # sliding in horizontal direction
    for i in range(X.shape[0]-f.shape[0]+1):
        # sliding in vertical direction
        for j in range(X.shape[1]-f.shape[1]+1):
            new_image.append(X[:,:,k][i:i+f.shape[0],j:j+f.shape[1]])
            
# resizing the generated patches as per number of images
new_image = np.array(new_image)
new_image.resize((X.shape[2],int(new_image.shape[0]/X.shape[2]),new_image.shape[1],new_image.shape[2]))
new_image.shape

# number of features in data set
s_row = X.shape[0] - f.shape[0] + 1
s_col = X.shape[1] - f.shape[1] + 1
num_filter = f.shape[2]

inputlayer_neurons = (s_row)*(s_col)*(num_filter)
output_neurons = 1 

# initializing weight
wo=np.random.uniform(size=(inputlayer_neurons,output_neurons))

# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# generating output of convolution layer
filter_output = []
# for each image
for i in range(len(new_image)):
    # apply each filter
    for k in range(f.shape[2]):
        # do element wise multiplication
        for j in range(new_image.shape[1]):
            filter_output.append((new_image[i][j]*f[:,:,k]).sum()) 

filter_output = np.resize(np.array(filter_output), (len(new_image),f.shape[2],new_image.shape[1]))

# applying activation over convolution output
filter_output_sigmoid = sigmoid(filter_output)

filter_output.shape, filter_output_sigmoid.shape

