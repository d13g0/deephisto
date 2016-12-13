import random, sys, pdb
import caffe
import numpy as np
from PIL import Image

from caffe_settings import CaffeSettings

# Making sure that this file is visible to caffe
sys.path.insert(0, CaffeSettings.CAFFE_CODE_DIR)


class DataLayer(caffe.Layer):
    """
    Loads the input,label image pairs from our dataset

    NOTE: The directory containing this class must be added manually to your PYTHONPATH variable

    e.g. export PYTHONPATH=$PYTHONPATH:/<some-path>/deephisto/caffe
    """

    def setup(self, bottom, top):
        if len(top) != 2:
            raise AssertionError('Two tops are needed: data and label')
        if len(bottom) != 0:
            raise AssertionError('This is a source layer. No bottoms required')

        params = eval(self.param_str)

        self.data_dir = params['data_dir']
        self.split_dir = params['split_dir']
        self.stage = params['stage']

        self.random = params.get('random', True)
        self.seed = params.get('seed', None)
        self.angle = 0

        avg_img_file = self.split_dir + '/' + CaffeSettings.AVG_IMG
        self.mean = np.array(Image.open(avg_img_file))  # CaffeLocations.TRAINING_MEAN #

        datafile = '%s/%s.txt' % (self.split_dir, self.stage)  # stage = (training, validation)

        self.indices = open(datafile, 'r').read().splitlines()
        self.idx = 0

        if self.stage != CaffeSettings.STAGE_TRAIN:
            self.random = False

        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices) - 1)

        print
        print 'Data Layer setup'
        print '-------------------------------------------------'
        print 'Random  selection : %s' % self.random
        print 'Stage             : %s' % self.stage
        print '-------------------------------------------------'
        print
        # raw_input('<< Press any key to continue >>')

    def reshape(self, bottom, top):
        imgfile, labelfile = self.indices[self.idx].split(';')
        self.data = self.load_image(imgfile)
        self.label = self.load_label(labelfile)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                print  '-------------------------------------'
                print
                print  'LOOPING THROUGH THE %s DATASET' % self.stage
                print
                print  '-------------------------------------'
                self.idx = 0

        #self.idx = 1251 #testing the loss layer

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, filename):
        img = Image.open('%s/%s' % (self.data_dir, filename))
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
        img = Image.open('%s/%s' % (self.data_dir, filename))
        label = np.array(img, dtype=np.uint8)
        label = label[:, :, 0]  # take any channel (flatten png to grayscale image)
        label = label[np.newaxis, ...]  # add the batch dimension
        return label
