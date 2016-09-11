
import glob, pdb
import numpy as np


import os.path
import matplotlib.pyplot as plt
import caffe

from PIL import Image

# Make sure that caffe is on the python path:
deephisto_root = '/home/dcantor/projects/deephisto/code'
import sys,pdb
sys.path.insert(0, deephisto_root + '/caffe')

MODEL_FILE = '/home/dcantor/projects/deephisto/caffe/train.prototxt'
PRETRAINED = '/home/dcantor/projects/deephisto/caffe/s1/_iter_16.caffemodel'
PATCHES_DIR = '/home/dcantor/projects/deephisto/patches'
IMAGE_FILE = None
LABEL_FILE = None
net = None

def load():
    global net
    caffe.set_mode_cpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

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

def set_files():
    global IMAGE_FILE, LABEL_FILE
    sources = glob.glob(PATCHES_DIR + '/' + 'P_*_MU_*.png')
    N = len(sources)
    idx = np.random.randint(0, N - 1)

    IMAGE_FILE = sources[idx]
    LABEL_FILE = IMAGE_FILE.replace('MU','HI')

    print
    print 'Files'
    print '---------------------------------------'
    print 'Index : ' + str(idx)
    print 'Input : ' + os.path.basename(IMAGE_FILE)
    print 'Label : ' + os.path.basename(LABEL_FILE)
    print


def send_image():

    set_files()
    input_image = np.array(Image.open(IMAGE_FILE))
    print 'input image shape ' , input_image.shape
    mean = np.array([121, 82, 82], dtype=np.float32)
    img = np.array(input_image, dtype=np.float32)
    img -= mean
    img = img[:,:,::-1]
    img = img.transpose((2,0,1))

    net.blobs['data'].data[...] = img
    print net.blobs['data'].data.shape
    out = net.forward()

def get():
    send_image()

    input_image = np.array(Image.open(IMAGE_FILE))
    fig,ax = plt.subplots()
    fig.canvas.set_window_title('Input Image :'+os.path.basename(IMAGE_FILE))
    plt.imshow(input_image, interpolation='none')

    im = np.array(Image.open(LABEL_FILE))
    im = im[:, :, 0]
    fig, ax2 = plt.subplots()
    fig.canvas.set_window_title('Label Image :' + os.path.basename(LABEL_FILE))
    img2 = plt.imshow(im, interpolation='None', cmap='jet', vmin=0, vmax=20)
    ax2.format_coord = get_formatter(im)
    fig.colorbar(img2)

    s= net.blobs['score'].data
    s = s[0,...] #remove the batch dimension
    s = s.transpose(2,1,0) #change to height x width x channel   (not sure about the order here for height and width)
    s = s.sum(axis=2)  #collapse image summing over all labels
    fig,ax = plt.subplots()
    fig.canvas.set_window_title('Output Image ')
    img = ax.imshow(s, interpolation='none')
    fig.colorbar(img)
    ax.format_coord = get_formatter(s)



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
