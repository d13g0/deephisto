

import os
os.environ['GLOG_minloglevel'] = '1'

import glob, pdb
import numpy as np
from itertools import product

import os.path
import matplotlib.pyplot as plt
import caffe

from PIL import Image
from NetworkDefinitions import NetworkDefinitions


# Make sure that caffe is on the python path:
deephisto_root = '/home/dcantor/projects/deephisto/code'
import sys,pdb
sys.path.insert(0, deephisto_root + '/caffe')

MODEL_FILE = '/home/dcantor/projects/deephisto/caffe/deploy.prototxt'
SNAPSHOT = '/home/dcantor/projects/deephisto/caffe/%s/_iter_%d.caffemodel'
PATCHES_DIR = '/home/dcantor/projects/deephisto/patches'
IMAGE_FILE = None
LABEL_FILE = None
net = None
EPOCH = None

def load(directory, snapshot):
    global net, EPOCH
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    EPOCH = int(snapshot)
    weights = SNAPSHOT%(directory,EPOCH)


    net = caffe.Net(MODEL_FILE, caffe.TEST, weights=weights)
    print 'snapshot ' + weights + ' loaded.'

def show():
    print '----------------------------------------------------------------'
    print 'NETWORK STRUCTURE'
    print '----------------------------------------------------------------'
    print
    print 'Layers'
    print
    for layer_name, blob in net.blobs.iteritems():
        hasdata = np.any(blob.data>0)
        print layer_name.ljust(20) + '\t' + str(blob.data.shape).ljust(15) +'\t' + ('OK' if hasdata else 'NO DATA')
    print
    print 'Parameter Shapes (weights) (biases)'
    print
    for layer_name, param in net.params.iteritems():
        hasdata = np.any(param[0].data>0)
        try:
            print layer_name.ljust(20) + '\t' + str(param[0].data.shape).ljust(15), str(param[1].data.shape) +'\t' +('OK' if hasdata else 'NO DATA')
        except IndexError:
            pass
    print
    print

def set_files(patch=None):
    global IMAGE_FILE, LABEL_FILE
    sources = glob.glob(PATCHES_DIR + '/' + 'P_*_MU_*.png')

    if patch is None:
        N = len(sources)
        idx = np.random.randint(0, N - 1)

        IMAGE_FILE = sources[idx]
    else:
        if not patch.endswith('.png'):
            patch = patch + '.png'
        list = [f for f in sources if os.path.basename(f) == patch]
        if len(list) == 0:
            raise Exception('Image %s does not exist'%patch)
        else:
            IMAGE_FILE = list[0]
            idx = None

    LABEL_FILE = IMAGE_FILE.replace('MU','HI')

    print
    print 'Files'
    print '---------------------------------------'
    print 'Index : ' + str(idx)
    print 'Input : ' + os.path.basename(IMAGE_FILE)
    print 'Label : ' + os.path.basename(LABEL_FILE)
    print


def send_image(patch=None):

    set_files(patch)
    input_image = np.array(Image.open(IMAGE_FILE))
    img = np.array(input_image, dtype=np.float32)
    img -= NetworkDefinitions.TRAINING_MEAN   #subtract mean
    img = img[:,:,::-1]  #RGB -> BGR
    img = img.transpose((2,1,0)) #transpose to channel x height x width
    net.blobs['data'].data[...] = img

    net.forward()

    prediction = net.blobs['score'].data
    prediction = prediction[0,...] #remove the batch dimension
    prediction = prediction.argmax(axis=0)
    return prediction

def sampler():
    import matplotlib.pylab as plt
    from matplotlib import gridspec

    GSIZE = 3
    fig, ax = plt.subplots(GSIZE*2,GSIZE*2, figsize=(12,12))


    fig.canvas.set_window_title('DeepHisto Sampler [epoch: %d]'%EPOCH)
    #fig.suptitle('DeepHisto Sampler', fontsize=18, color='white')

    for i,j in product(range(0,GSIZE*2),range(0,GSIZE)):
        pred = send_image()
        label = np.array(Image.open(LABEL_FILE))
        label = label[:, :, 0]
        if (i==0):
            ax[0, 2 * j].set_title('GT')
            ax[0, 2 * j + 1].set_title('PR')

        ax[i, 2 * j ].imshow(label,interpolation='None', cmap='jet', vmin=0, vmax=NetworkDefinitions.NUM_LABELS)
        ax[i, 2 * j + 1].imshow(pred,interpolation='None', cmap='jet', vmin=0, vmax=NetworkDefinitions.NUM_LABELS)
        ax[i, 2 * j].get_xaxis().set_visible(False)
        ax[i, 2 * j].get_yaxis().set_visible(False)
        ax[i, 2 * j +1].get_xaxis().set_visible(False)
        ax[i, 2 * j +1].get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()




def get(patch=None):

    prediction = send_image(patch)

    image = np.array(Image.open(IMAGE_FILE))
    label = np.array(Image.open(LABEL_FILE))
    label = label[:, :, 0]

    fig,ax = plt.subplots(1,3, figsize=(12,5))

    fig.canvas.set_window_title(os.path.basename(IMAGE_FILE))

    ax[0].set_title('Input')
    ax[0].imshow(image, interpolation='none')

    ax[1].set_title('Label')
    img2 = ax[1].imshow(label, interpolation='None', cmap='jet', vmin=0, vmax=NetworkDefinitions.NUM_LABELS)
    ax[1].format_coord = get_formatter(label)


    ax[2].set_title('Output Image')
    img3 = ax[2].imshow(prediction, interpolation='none', vmin=0, vmax=NetworkDefinitions.NUM_LABELS)
    ax[2].format_coord = get_formatter(prediction)

    plt.tight_layout()
    plt.show()


def get_formatter(img):

    img = img
    def formatter(x,y):
        numcols, numrows = img.shape
        row = int(x + 0.5)
        col = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = img[col, row]
            return 'x=%d, y=%d, Value =%.2f' % (col, row, z)
        else:
            return 'x=%d, y=%d Outside' % (y, x)

    return formatter

def go():
    load()
    get()

if __name__=='__main__':
    pdb.set_trace()
