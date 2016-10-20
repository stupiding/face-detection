import sys
sys.path.append('/home/guojinma/CNN_Tools/caffe/python')
import caffe
import numpy as np


class FaceRectLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance.")
        self.params_ = eval(self.param_str)

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        if bottom[0].num != bottom[2].num or bottom[2].num != bottom[2].count:
            raise Exception("The third input blob should be labels")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.labels = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data.reshape(bottom[0].data.shape)
        dim = bottom[0].count / bottom[0].num
        for d in xrange(dim):
            self.labels[:,d][...] = bottom[2].data.reshape([bottom[0].num, 1, 1])
        self.diff[...] = self.diff * self.labels
        if self.params_['norm'] is 'smoothL1':
            def smoothL1(x):
                if abs(x) < 1:
                    return 0.5 * x * x
                else:
                    return abs(x) - 0.5
            top[0].data[...] = sum([smoothL1(x) for x in self.diff]) / np.sum(bottom[2].data)
        elif self.params_['norm'] is 'L1':
            top[0].data[...] = np.sum(np.abs(self.diff)) / np.sum(bottom[2].data)
        elif self.params_['norm'] is 'L2':
            top[0].data[...] = np.sum(self.diff**2) / np.sum(bottom[2].data) / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if self.params_['norm'] is 'L1':
                self.diff[...] = np.ones_like(bottom[0].data, dtype = np.float32)
                if i == 0:
                    self.diff[self.diff >= 0] = -1
                else :
                    self.diff[self.diff < 0] = -1
            elif self.params_['norm'] is 'smoothL1':
                self.diff[...] = np.ones_like(bottom[0].data, dtype = np.float32)
            elif self.params_['norm'] is 'L2':
                if not propagate_down[i]:
                    continue
                if i == 0:
                    sign = 1
                else:
                    sign = -1
                bottom[i].diff[...] = sign * self.diff / bottom[i].num
