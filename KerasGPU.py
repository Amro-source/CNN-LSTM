# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:11:44 2020

@author: Zikantika
"""

import keras
import tensorflow as tf


config =tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)