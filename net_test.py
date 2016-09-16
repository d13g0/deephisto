
import os
os.environ['GLOG_minloglevel'] = '2'
import pdb
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

import deephisto.caffe
from deephisto import ImageUtils, Locations, PatchSampler
from deephisto.caffe.CaffeLocations import CaffeLocations

class SliceTest:

    def __init__(self, locations):

        self.utils = ImageUtils(locations)
        self.subject = None
        self.index = None
        self.mean = np.array(Image.open(CaffeLocations.AVG_IMG))


    def load_data(self, subject,index):
        self.subject = subject
        self.index = index

        utils = self.utils
        utils.set_subject(subject)

        self.mask = utils.load_mask_png(index)
        self.target = Image.fromarray(np.zeros(shape=self.mask.shape[0:2]))
        self.bmask = utils.get_binary_mask(self.index)
        self.hist = utils.load_histo_png_image(index)
        self.hist_flat = self.hist[:, :, 0]
        self.input = utils.load_multichannel_input(index)
        self.input = self.input.astype(np.uint8)
        self.input_image = Image.fromarray(self.input)

    def load_network(self,directory, epoch):
        deephisto.caffe.set_mode_gpu()
        self.directory = directory
        self.epoch = epoch
        model = CaffeLocations.SNAPSHOT_DIR % (directory, epoch)
        print 'Loading %s' % model
        self.net = deephisto.caffe.Net(CaffeLocations.DEPLOY_NET_PROTOTXT, deephisto.caffe.TEST, weights=model)

    def init_show(self):
        fig, ax = plt.subplots(1, 3, facecolor='black', figsize=(14, 4))
        fig.canvas.set_window_title('%s %s' % (self.subject, self.index))

        for x in ax:
            x.get_xaxis().set_visible(False)
            x.get_yaxis().set_visible(False)

        ax[0].set_title('input', color='white')
        ax[0].imshow(self.input, interpolation='None')

        ax[1].set_axis_bgcolor('black')
        ax[1].set_title('ground truth', color='white')
        ax[1].imshow(self.hist_flat, vmin=0, vmax=10, cmap='jet', interpolation='None')

        #ax[2].imshow(self.mask, alpha=0.5)
        nothing = np.zeros(shape=self.mask.shape)
        self.plot_img = ax[2].imshow(nothing,interpolation='None',cmap='jet',vmin=0,vmax=10)

        ax[2].set_axis_bgcolor('black')
        ax[2].set_title('prediction', color='white')


        self.fig = fig
        self.ax = ax
        plt.draw()
        plt.subplots_adjust(left=0, bottom=0, right=1, wspace=0, hspace=0)

    def update_show(self,x,y,data):
        fig = self.fig
        ax = self.ax
        pimage = Image.fromarray(data.astype(np.uint8))
        bx = x-dh.Sampler.WSIDE
        by = y-dh.Sampler.WSIDE
        self.target.paste(pimage,(bx,by))
        self.plot_img.set_data(np.array(self.target, dtype=np.uint8))

        plt.pause(0.01)



    def get_patch(self,image, x,y):
        Sampler = Sampler
        cimage = image.crop((x - Sampler.WSIDE, y - Sampler.WSIDE, x + Sampler.WSIDE, y + Sampler.WSIDE))
        return np.array(cimage)

    def get_pair(self, x,y):
        input = self.get_patch(self.input_image, x, y)
        label = self.get_patch(Image.fromarray(self.hist_flat), x, y)
        return input, label


    def preprocess_input(self, input):
        #process input
        img = input.astype(np.float32)
        img -= self.mean
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.transpose((2, 1, 0))  # transpose to channel x height x width
        return img

    def init_plot(self):
        fig, ax = plt.subplots(1, 3)
        for axis in ax:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        self.fig = fig
        self.ax = ax

    def show_patch(self,x,y):
        input, label = self.get_pair(x,y)
        pred = self.predict(input, label)
        #fig, ax = plt.subplots(1,3)
        # for axis in ax:
        #     axis.get_xaxis().set_visible(False)
        #     axis.get_yaxis().set_visible(False)
        fig = self.fig
        ax = self.ax

        fig.canvas.set_window_title('DeepHisto Patch %d,%d'%(x,y))

        ax[0].set_title('Input')
        ax[0].imshow(input, interpolation='None')
        ax[1].set_title('Ground Truth')
        ax[1].imshow(label,interpolation='None', vmin=0, vmax=10, cmap='jet')
        ax[2].set_title('Prediction')
        ax[2].imshow(pred, interpolation='None', vmin=0, vmax=10, cmap='jet')
        #plt.tight_layout()
        plt.pause(0.1)

    def predict(self, input, label):
        net = self.net
        caffe_input = self.preprocess_input(input)
        net.blobs['data'].data[...] = caffe_input
        net.forward()
        pred = net.blobs['score'].data
        pred = pred[0, ...]
        pred = pred.argmax(axis=0)
        return pred

    def go(self):
        #
        # Cartesian convention used
        # x = columns
        # y = rows
        # ----------------> x
        # |
        # |
        # |
        # v y
        # In numpy first dimension is always rows!

        y = 0  # rows   cartesian vs. image convention
        x = 0 #columns  cartesian vs. image convention

        #numpy indexing shape returns rows, columns thus calling shape returns --> height, width
        MAX_Y, MAX_X = self.hist_flat.shape
        STEP = PatchSampler.WSIZE

        print 'One moment please ....'
        #self.init_plot()
        self.init_show()

        while (x <= MAX_X and y <= MAX_Y):
             while (x < MAX_X):
                 input, label = self.get_pair(x,y)
                 pred = self.predict(input,label)
                 self.update_show(x,y, pred)
                 #self.show_patch(x,y)

                 x = x + STEP
             y = y + STEP
             x = 0
        plt.show()

if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    pr = SliceTest(locations)
    pr.load_data('EPI_P036',1)
    pr.load_network('wed',16000)
    pr.go()
