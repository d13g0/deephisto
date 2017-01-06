#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

import numpy as np
import random, sys, pdb, os
from PIL import Image

import caffe


#import matplotlib.pylab as plt

# adds the directory where this script resides so net can load the DataLayer
# @see: https://github.com/rbgirshick/py-faster-rcnn/issues/98
sys.path.insert(0,os.path.dirname(__file__))




class DataLayer(caffe.Layer):
    """
    This layer feeds data to a caffe network for training and validation

    ATTENTION!  The directory containing this class must be added manually to your PYTHONPATH variable

    e.g. export PYTHONPATH=$PYTHONPATH:/<some-path>/deephisto/net
    """

    ROTATIONS = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]


    def setup(self, bottom, top):
        if len(top) != 2:
            raise AssertionError('Two tops are needed: data and label')
        if len(bottom) != 0:
            raise AssertionError('This is a source layer. No bottoms required')


        params = eval(self.param_str)
        self.PATCH_DIR = params['patch_dir']
        self.DATASET_DIR = params['dataset_dir']
        self.AVERAGE_IMAGE = params['average_image']
        self.STAGE = params['stage']
        self.RANDOM_TRAINING = params['random_training']
        self.INCLUDE_ROTATIONS = params['include_rotations']
        self.NUM_ROTATIONS = len(DataLayer.ROTATIONS)

        self.mean = np.array(Image.open(self.AVERAGE_IMAGE))

        if self.STAGE == 'training':
            patch_list = params['training_patches']
        elif self.STAGE == 'validation':
            patch_list = params['validation_patches']

        self.indices = open(patch_list, 'r').read().splitlines()
        self.idx = 0

        if self.STAGE != 'training':
            self.RANDOM_TRAINING = False

        if self.RANDOM_TRAINING:
            self.idx = random.randint(0, len(self.indices) - 1)

        print
        print 'Data Layer setup'
        print '-------------------------------------------------'
        print 'Stage             : %s' % self.STAGE
        print 'Random  selection : %s' % self.RANDOM_TRAINING
        print 'Include rotations : %s' % self.INCLUDE_ROTATIONS
        print '-------------------------------------------------'
        print


    def reshape(self, bottom, top):
        imgfile, labelfile = self.indices[self.idx].split(';')

        if self.INCLUDE_ROTATIONS:
            self.rotation = self.compute_rotation()

        self.data = self.load_image(imgfile)
        self.label = self.load_label(labelfile)

        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.RANDOM_TRAINING:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                print  '-------------------------------------'
                print
                print  'LOOPING THROUGH THE %s DATASET' % self.STAGE
                print
                print  '-------------------------------------'
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def compute_rotation(self):
        rot = random.randint(0, self.NUM_ROTATIONS)
        if rot == self.NUM_ROTATIONS:
            return None
        else:
            return DataLayer.ROTATIONS[rot]




    def load_image(self, filename):
        img = Image.open('%s/%s' % (self.PATCH_DIR, filename))

        #-------------------------------------------------------
        if self.INCLUDE_ROTATIONS and self.rotation is not None:
            img = img.transpose(self.rotation)
        # ------------------------------------------------------

        img = np.array(img, dtype=np.float32)
        img -= self.mean  # subtract mean value
        img = img[:, :, ::-1]  # switch channels RGB -> BGR
        img = img.transpose((2, 0, 1))  # transpose to channel x height x width
        return img

    def load_label(self, filename):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        img = Image.open('%s/%s' % (self.PATCH_DIR, filename))

        # ------------------------------------------------------
        if self.INCLUDE_ROTATIONS and self.rotation is not None:
            img = img.transpose(self.rotation)
        # ------------------------------------------------------

        label = np.array(img, dtype=np.uint8)
        label = label[:, :, 0]  # take any channel (flatten png to grayscale image)
        label = label[np.newaxis, ...]  # add the batch dimension
        return label
