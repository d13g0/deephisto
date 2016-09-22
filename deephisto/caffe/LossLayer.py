import pdb
import numpy as np

import caffe


class TopoLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) !=2:
            raise Exception('We need two inputs to compute loss')

    def reshape(self, bottom, top):
        pdb.set_trace()

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass
