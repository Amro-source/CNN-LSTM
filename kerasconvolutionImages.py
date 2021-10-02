# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:33:56 2020

@author: Zikantika
"""
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

# build the model 
model = Sequential() 
model.add(Conv2D(32, kernel_size =(5, 5), strides =(1, 1), 
                 activation ='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2), strides =(2, 2))) 
model.add(Conv2D(64, (5, 5), activation ='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Flatten()) 
model.add(Dense(1000, activation ='relu')) 
model.add(Dense(num_classes, activation ='softmax')) 
  
# training the model 
model.compile(loss = keras.losses.categorical_crossentropy, 
              optimizer = keras.optimizers.SGD(lr = 0.01), 
              metrics =['accuracy']) 
  
# fitting the model 
model.fit(x_train, y_train, 
          batch_size = batch_size, 
          epochs = epochs, 
          verbose = 1, 
          validation_data =(x_test, y_test), 
          callbacks =[history]) 
  
# evaluating and printing results 
score = model.evaluate(x_test, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1]) 