import os

os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as ppa
from PIL import Image
import caffe

from deephisto import ImageUtils, Locations, PatchSampler
from deephisto.caffe import CaffeLocations


class NetTest:
    def __init__(self, locations):
        caffe.set_mode_gpu()
        self.utils = ImageUtils(locations)
        self.subject = None
        self.index = None
        self.WSIZE = None

    def set_window(self, wsize):

        self.WSIZE = wsize
        self.WSIDE = int(self.WSIZE / 2)

    def load_data(self, subject, index):

        self.subject = subject
        self.index = index

        utils = self.utils
        utils.set_subject(subject)

        try:
            self.mask = utils.load_mask_png(index)
        except IOError:
            raise ValueError('The slice %d does not exist or has not been annotated' % index)

        self.target = Image.fromarray(np.zeros(shape=self.mask.shape[0:2]))
        self.bmask = utils.get_binary_mask(self.index)

        self.hist = utils.load_unscaled_histo_png_image(index)
        self.hist_flat = self.hist[:, :, 0]

        self.input = utils.load_multichannel_input(index)
        self.input = self.input.astype(np.uint8)
        self.input_image = Image.fromarray(self.input)

    def load_network(self, directory, epoch, split_dir):

        self.directory = directory
        self.epoch = epoch

        model = CaffeLocations.SNAPSHOT_DIR % (directory, epoch)
        print 'Loading %s' % model

        #load network from 'directory'
        net_def = CaffeLocations.NET_DIR + '/' + directory + '/' + CaffeLocations.DEPLOY_PROTO
        self.net = caffe.Net(net_def, caffe.TEST, weights=model)

        #load mean to preprocess samples form 'data_dir'
        avg_img_file = CaffeLocations.SPLIT_DIR + '/' + split_dir + '/' + CaffeLocations.AVG_IMG
        self.mean = np.array(Image.open(avg_img_file))



    def setup(self):

        if self.WSIZE is None:
            raise 'Please set the window size first'

        plt.ion()
        fig, ax = plt.subplots(1, 3, facecolor='black', figsize=(14, 4), sharex=True, sharey=True)
        fig.canvas.set_window_title('%s %s' % (self.subject, self.index))

        for x in ax:
            x.get_xaxis().set_visible(False)
            x.get_yaxis().set_visible(False)

        ax[0].set_axis_bgcolor('black')
        ax[0].set_title('input', color='white')
        ax[0].imshow(self.input, interpolation='None')

        ax[1].set_axis_bgcolor('black')
        ax[1].set_title('ground truth', color='white')
        ax[1].imshow(self.hist_flat, vmin=0, vmax=10, cmap='jet', interpolation='None')

        ax[2].set_axis_bgcolor('black')
        ax[2].set_title('prediction', color='white')
        self.plot_img = ax[2].imshow(np.zeros(shape=self.mask.shape), interpolation='None', cmap='jet', vmin=0, vmax=10)

        self.rect = ppa.Rectangle((0, 0), self.WSIZE, self.WSIZE, linewidth=1, edgecolor='r',
                                  facecolor='none')
        ax[2].add_patch(self.rect)

        self.fig = fig
        self.ax = ax
        plt.draw()
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, wspace=0, hspace=0)

    def update(self, x, y, data):
        if self.WSIDE is None:
            raise 'Please set the window size first'
        fig = self.fig
        ax = self.ax

        bimage = self.blend(x,y,data)

        pimage = Image.fromarray(bimage.astype(np.uint8))
        bx = x - self.WSIDE
        by = y - self.WSIDE
        self.target.paste(pimage, (bx, by))

        self.plot_img.set_data(np.array(self.target, dtype=np.uint8))

        #plt.pause(0.00001)
        # plt.draw()

    def blend(self, x,y, pred):
        base = self.get_patch(self.target, x,y)
        assert base.shape == pred.shape
        return (base + pred)/2


    def get_patch(self, image, x, y):
        if self.WSIDE is None:
            raise 'Please set the window size first'
        cimage = image.crop(
                (x - self.WSIDE, y - self.WSIDE, x + self.WSIDE, y + self.WSIDE))
        return np.array(cimage)

    def get_sample(self, x, y):
        input = self.get_patch(self.input_image, x, y)
        label = self.get_patch(Image.fromarray(self.hist_flat), x, y)
        return input, label

    def preprocess_input(self, input):
        # process input
        img = input.astype(np.float32)
        img -= self.mean
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.transpose((2, 0, 1))  # transpose to channel x height x width
        return img

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
        if self.WSIZE is None or self.WSIDE is None:
            raise 'Please set the window size first'

        self.setup()

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
        x = 0  # columns  cartesian vs. image convention

        # numpy indexing shape returns rows, columns thus calling shape returns --> height, width
        MAX_Y, MAX_X = self.hist_flat.shape
        STEP = self.WSIZE/4

        print 'One moment please ....'


        while (y <= MAX_Y):
            while (x < MAX_X):
                input, label = self.get_sample(x, y)
                pred = self.predict(input, label)
                self.update(x, y, pred)
                x = x + STEP
                self.rect.set_x(x - self.WSIDE)
                self.rect.set_y(y - self.WSIDE)
            x = 0
            plt.pause(0.0001)
            y = y + STEP
        plt.show(block=True)


if __name__ == '__main__':
    pr = NetTest(Locations('/home/dcantor/projects/deephisto'))
    pr.set_window(28)
    pr.load_network('dh28', 300000, '28x28')
    pr.load_data('EPI_P036', 3)
    pr.go()
