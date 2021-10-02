


import onnx
from onnx import backend
from onnx2keras import onnx_to_keras

#import caffe2.python.onnx.backend
#from caffe2.python import core, workspace

import numpy as np

# make input Numpy array of correct dimensions and type as required by the model

onnx_model = onnx.load('netflowers.onnx')

k_model = onnx_to_keras(onnx_model, ['input'])

#output = caffe2.python.onnx.backend.run_model(modelFile, inputArray.astype(np.float32))