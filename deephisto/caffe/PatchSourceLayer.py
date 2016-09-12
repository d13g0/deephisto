import caffe
import numpy as np
from PIL import Image
import random
import pdb
from NetworkDefinitions import NetworkDefinitions


class PatchSourceLayer(caffe.Layer):
    """
    Loads the input,label image pairs from our dataset

    NOTE: The directory containing this class must be added manually to your PYTHONPATH variable

    e.g. export PYTHONPATH=$PYTHONPATH:/<some-path>/deephisto/caffe
    """

    def setup(self, bottom, top):
        if len(top) !=2:
            raise AssertionError('Two tops are needed: data and label')
        if len(bottom) !=0:
            raise AssertionError('This is a source layer. No bottoms required')

        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.stage = params['stage']
        self.random = params.get('randomize',True)
        self.seed = params.get('seed', None)

        #look /deephisto/patches/training_average.txt
        self.mean = NetworkDefinitions.TRAINING_MEAN

        datafile = '%s/%s.txt' % (self.data_dir, self.stage)  # stage = (training, validation)
        self.indices = open(datafile, 'r').read().splitlines()
        self.idx = 0

        if self.stage != 'training':
            self.random = False

        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom,top):
        imgfile,labelfile = self.indices[self.idx].split(';')
        self.data = self.load_image(imgfile)
        self.label = self.load_label(labelfile)

        top[0].reshape(1, *self.data.shape)

        top[1].reshape(1, *self.label.shape)



    def forward(self,bottom,top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        if self.random:
            self.idx = random.randint(0,len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx =0


    def backward(self,top, propagate_down, bottom):
        pass


    def load_image(self, filename):
        #print filename
        img = Image.open('%s/%s'%(self.data_dir,filename))
        img = np.array(img, dtype=np.float32)
        img -= self.mean #subtract mean value
        img = img[:,:,::-1]  #switch channels RGB -> BGR
        ###i#mg = img[:,:,0:2] #removing MD info
        ###img = img[:, :, 0:1] # only consider MRI
        img = img.transpose((2,1,0))  #transpose to channel x height x width
        return img


    def load_label(self, filename):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        img = Image.open('%s/%s' % (self.data_dir, filename))
        label = np.array(img, dtype=np.uint8)
        label = label[:,:,0]  #take any channel (flatten png to grayscale image)
        #label[np.where(label>2)] =5 # this must be easier to predict
        #label[np.where(label<=2)] = 0 # binarize output
        label = label[np.newaxis, ...] #add the batch dimension
        return label