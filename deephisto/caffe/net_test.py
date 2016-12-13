import os,pdb

os.environ['GLOG_minloglevel'] = '2'
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.patches as ppa
from PIL import Image
import caffe

from deephisto import ImageUtils, Locations, PatchSampler
from deephisto.caffe import CaffeSettings


class NetTest:

    BLEND_MEAN = 'avg'
    BLEND_MAX = 'max'
    BLEND_MEDIAN = 'med'
    BLEND_MODE  = 'mod'
    BLEND_MODE_MAX = 'mod+max'

    BLEND_MODES = [BLEND_MEAN, BLEND_MAX, BLEND_MEDIAN, BLEND_MODE, BLEND_MODE_MAX ]


    def __init__(self, locations):
        caffe.set_mode_gpu()
        self.utils = ImageUtils(locations)
        self.subject = None
        self.index = None
        self.WSIZE = None
        self.BMODE = NetTest.BLEND_MEAN

    def set_window(self, wsize):

        self.WSIZE = wsize
        self.WSIDE = int(self.WSIZE / 2)

    def set_blend(self, blend_mode):
        if blend_mode not in NetTest.BLEND_MODES:
            raise AssertionError('Blend mode is not identified [%s]'%blend_mode)
        self.BMODE = blend_mode
        print 'Blend mode set to :%s'%self.BMODE

    def load_data(self, subject, index):

        self.subject = subject
        self.index = index

        utils = self.utils
        utils.set_subject(subject)

        try:
            self.mask = utils.load_mask_for_slice(index)
        except IOError:
            raise ValueError('The slice %d does not exist or has not been annotated' % index)

        self.target = Image.fromarray(np.zeros(shape=self.mask.shape[0:2]))
        self.bmask = utils.load_binary_mask_for_slice(self.index)

        self.hist = utils.load_labels_for_slice(index)
        self.hist_flat = self.hist[:, :, 0]

        self.input = utils.load_multichannel_input(index)
        self.input = self.input.astype(np.uint8)
        self.input_image = Image.fromarray(self.input)

    def load_network(self, directory, epoch, split_dir):

        self.directory = directory
        self.epoch = epoch

        model = CaffeSettings.SNAPSHOT_DIR % (directory, epoch)
        print 'Loading %s' % model

        #load network from 'directory'
        net_def = CaffeSettings.NET_DIR + '/' + directory + '/' + CaffeSettings.DEPLOY_PROTO
        self.net = caffe.Net(net_def, caffe.TEST, weights=model)

        #load mean to preprocess samples form 'data_dir'
        avg_img_file = CaffeSettings.SPLIT_DIR + '/' + split_dir + '/' + CaffeSettings.AVG_IMG
        self.mean = np.array(Image.open(avg_img_file))



    def setup(self):

        if self.WSIZE is None:
            raise 'Please set the window size first'

        plt.ion()
        fig, ax = plt.subplots(1, 3, facecolor='black', figsize=(14, 4), sharex=True, sharey=True)
        fig.canvas.set_window_title('%s - %d Subject: %s  [%s]  (%s)' % (self.directory, self.epoch, self.subject, self.index, self.BMODE.upper()))

        for x in ax:
            x.get_xaxis().set_visible(False)
            x.get_yaxis().set_visible(False)

        ax[0].set_axis_bgcolor('black')
        ax[0].set_title('input', color='white')
        ax[0].imshow(self.input, interpolation='None')

        ax[1].set_axis_bgcolor('black')
        ax[1].set_title('ground truth', color='white')
        ax[1].imshow(self.hist_flat, cmap='jet', interpolation='None', vmin=0, vmax=CaffeSettings.NUM_LABELS)
        ax[1].format_coord = self._get_formatter('Ground Truth', self.hist_flat)

        ax[2].set_axis_bgcolor('black')
        ax[2].set_title('prediction', color='white')
        self.plot_img = ax[2].imshow(np.zeros(shape=self.mask.shape), interpolation='None', cmap='jet', vmin=0, vmax=CaffeSettings.NUM_LABELS)


        self.rect = ppa.Rectangle((0, 0), self.WSIZE, self.WSIZE, linewidth=1, edgecolor='#ff0000', facecolor='none', alpha=0.5)
        ax[2].add_patch(self.rect)

        self.fig = fig
        self.ax = ax
        plt.pause(0.0001)
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, wspace=0, hspace=0)

    def new_pass(self):
        self.target = Image.fromarray(np.zeros(shape=self.mask.shape[0:2]))
        self.plot_img = self.ax[2].imshow(np.zeros(shape=self.mask.shape), interpolation='None', cmap='jet', vmin=0, vmax=CaffeSettings.NUM_LABELS)

    def update_pass(self, x, y, data):

        if self.WSIDE is None:
            raise 'Please set the window size first'
        fig = self.fig
        ax = self.ax

        if (data.min() ==0 and data.max() ==0):
            return

        patch = Image.fromarray(data.astype(np.float))
        bx = x - self.WSIDE
        by = y - self.WSIDE
        self.target.paste(patch, (bx, by))
        self.plot_img.set_data(np.array(self.target, dtype=np.float))



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


    def blend(self, x,y, pred):

        base = self.get_patch(self.target, x,y)

        if (base.min()==0 and base.max()==0):
            return pred

        assert base.shape == pred.shape

        if self.BMODE == NetTest.BLEND_MEAN:
            return np.mean([base, pred], axis=0)

        elif self.BMODE == NetTest.BLEND_MAX:
            return np.maximum(base, pred)

        elif self.BMODE == NetTest.BLEND_MEDIAN:
            return pred #np.median([base, pred], axis=0)#pred #the median occurs after all the passes



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

        STEPS = 7
        STEP = int(self.WSIZE/ STEPS)

        print 'One moment please ....'
        plt.ion()

        image_pass = np.array(self.target, dtype=np.uint8)
        h,w = image_pass.shape
        passes = np.zeros(shape=(STEPS, h,w))

        idx = 0
        for offset in range(STEPS):

            self.new_pass()

            print 'pass %d of %d'%(offset+1, STEPS)
            self.ax[2].set_title('pass %d of %d'%(offset+1, STEPS), color='white')

            y = offset*STEP
            x = offset*STEP



            while (y <= MAX_Y):
                while (x < MAX_X):
                    #rect = ppa.Rectangle((x-self.WSIDE, y-self.WSIDE), self.WSIZE, self.WSIZE, linewidth=1, edgecolor='#dddddd', facecolor='none', alpha=0.1)
                    #self.ax[2].add_patch(rect)

                    input, label = self.get_sample(x, y)
                    pred = self.predict(input, label)
                    self.update_pass(x, y, pred)
                    x = x + self.WSIZE

                self.rect.set_x(x - self.WSIDE)
                self.rect.set_y(y - self.WSIDE)
                plt.pause(0.001)
                # plt.savefig('pass_%03d.png' % idx)
                # idx +=1
                x = offset*STEP
                y +=  self.WSIZE

            passes[offset] = np.array(self.target, dtype=np.float)


        plt.show(block=False)



        median = np.median(passes,axis=0)
        mode = stats.mode(passes, axis=0)[0][0]
        maximum = np.max(passes, axis=0)
        mean = np.mean(passes, axis=0)
        blend = np.copy(mode)
        blend[np.where(maximum >= 5)] = maximum[np.where(maximum >= 5)]


        if self.BMODE == NetTest.BLEND_MEDIAN:
            self.plot_img.set_data(median)
            self.ax[2].format_coord = self._get_formatter('Median',median)
            self.ax[2].set_title('Median', color='white')
        elif self.BMODE == NetTest.BLEND_MODE:
            self.plot_img.set_data(mode)
            self.ax[2].format_coord = self._get_formatter('Mode', mode)
            self.ax[2].set_title('Mode', color='white')
        elif self.BMODE == NetTest.BLEND_MAX:
            self.plot_img.set_data(maximum)
            self.ax[2].format_coord = self._get_formatter('Max', maximum)
            self.ax[2].set_title('Max', color='white')
        elif self.BMODE == NetTest.BLEND_MEAN:
            self.plot_img.set_data(mean)
            self.ax[2].format_coord = self._get_formatter('Mean', mean)
            self.ax[2].set_title('Mean', color='white')
        elif self.BMODE == NetTest.BLEND_MODE_MAX:
            self.plot_img.set_data(blend)
            self.ax[2].format_coord = self._get_formatter('Mode+Max', blend)
            self.ax[2].set_title('Mode + Max', color='white')


        plt.pause(0.001)


        fig, ax = plt.subplots(1,6,figsize=(12,3),sharex=True, sharey=True,facecolor='black')
        fig.canvas.set_window_title('%s - %d Subject: %s  [%s]' % (self.directory, self.epoch, self.subject, self.index))
        titles = ['Ground Truth','Median','Mode','Max','Mean','Mode + Max']
        images = [self.hist_flat,median,mode,maximum,mean, blend]

        for i in range(6):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].set_axis_bgcolor('black')
            ax[i].imshow(images[i], interpolation='None', cmap='jet', vmin=0, vmax=10)
            ax[i].set_title(titles[i],color='white', fontsize=16)
            ax[i].format_coord = self._get_formatter(titles[i], images[i])

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, wspace=0, hspace=0)
        plt.pause(0.001)


    def _get_formatter(self, title, img):

        img = img

        def formatter(x, y):
            numcols, numrows = img.shape
            row = int(x + 0.5)
            col = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = img[col, row]

                return '%s x,y: %d,%d, value: %.2f' % (title, col, row, z)
            else:
                return '%s x,y: %d,%d' % (title, col, row)

        return formatter


if __name__ == '__main__':
    pr = NetTest(Locations('/home/dcantor/projects/deephisto'))
    pr.set_window(28)
    pr.load_network('fcn8', 590000, '28x28b')
    pr.load_data('EPI_P040', 4)
    pr.set_window(28)
    pr.set_blend(NetTest.BLEND_MAX)
    pr.go()
    plt.show(block=True)
