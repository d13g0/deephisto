import os

os.environ['GLOG_minloglevel'] = '2'

import os.path
import sys
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import caffe
from PIL import Image
from CaffeLocations import CaffeLocations


# Making sure that this file is visible to caffe
sys.path.insert(0, CaffeLocations.CAFFE_CODE_DIR)

class NetTest:
    def __init__(self):

        caffe.set_mode_gpu()

        self.inputs = None
        self.labels = None
        self.predictions = None
        self.net = None
        self.directory = None
        self.epoch = None
        self.start = None
        self.end = None
        self.step = None
        self.fig = None
        self.ax = None
        self.panel_size = 3
        self.validation_data = None
        self.training_data = None
        self._load_validation()
        self.mean = np.array(Image.open(CaffeLocations.AVG_IMG))

    def _load_validation(self):
        datafile = CaffeLocations.SPLIT_DIR + '/validation.txt'
        with open(datafile, 'r') as dfile:
            lines = dfile.read().splitlines()
            self.validation_data = [f.split(';')[0] for f in lines]

        datafile = CaffeLocations.SPLIT_DIR + '/training.txt'
        with open(datafile, 'r') as dfile:
            lines = dfile.read().splitlines()
            self.training_data = [f.split(';')[0] for f in lines]

    def configure(self, directory, start, end, step):
        self.directory = directory
        self.start = start
        self.end = end
        self.step = step
        self.load_model(directory, start)
        print 'Observer configured'

    def load_model(self, directory, epoch):
        self.directory = directory
        self.epoch = epoch
        model = CaffeLocations.SNAPSHOT_DIR % (directory, epoch)
        print 'Loading %s' % model
        self.net = caffe.Net(CaffeLocations.DEPLOY_NET_PROTOTXT, caffe.TEST, weights=model)

    def show_network_model(self):
        net = self.net

        print 'NETWORK MODEL'
        print '----------------------------------------------------------------'
        print
        print 'Layers'
        print '------'
        print
        for layer_name, blob in net.blobs.iteritems():
            hasdata = np.any(blob.data > 0)
            print layer_name.ljust(20) + '\t' + str(blob.data.shape).ljust(15) + '\t' + ('OK' if hasdata else 'NO DATA')
        print
        print 'Parameters (weights) (biases)'
        print '-----------------------------'
        print
        for layer_name, param in net.params.iteritems():
            hasdata = np.any(param[0].data > 0)
            try:
                print layer_name.ljust(20) + '\t' + str(param[0].data.shape).ljust(15), str(
                        param[1].data.shape) + '\t' + ('OK' if hasdata else 'NO DATA')
            except IndexError:
                pass
        print
        print

    def get_files(self, patch_name=None, verbose=False):

        if patch_name is None:
            # if not specific patch is sought. Then get a random one from the validation list
            N = len(self.validation_data)
            idx = np.random.randint(0, N - 1)
            image_file = CaffeLocations.PATCHES_DIR + '/' + self.validation_data[idx]
        else:
            # retrieve the paths to the image and label files
            if not patch_name.endswith('.png'):
                patch_name = patch_name + '.png'

            #@TODO: Simplify this. This can be written succinctly
            list1 = [f for f in self.training_data if os.path.basename(f) == patch_name]
            list2 = [f for f in self.validation_data if os.path.basename(f) == patch_name]


            if len(list1) == 1:
                print '%s is a TRAINING example' % patch_name
                image_file = list1[0]
            elif len(list2) == 1:
                print '%s is a VALIDATION example' % patch_name
                image_file = list2[0]
            else:
                raise Exception('%s does not exist' % patch_name)

            image_file = CaffeLocations.PATCHES_DIR + '/' + image_file
            idx = None

        label_file = image_file.replace('MU', 'HI')

        if verbose:
            print
            print 'Files'
            print '---------------------------------------'
            print 'Index : ' + str(idx)
            print 'Input : ' + os.path.basename(image_file)
            print 'Label : ' + os.path.basename(label_file)
            print
        return image_file, label_file

    def preprocess_images(self, image_file, label_file):
        # process input image
        input_image = np.array(Image.open(image_file))
        img = np.array(input_image, dtype=np.float32)
        img -= self.mean
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.transpose((2, 1, 0))  # transpose to channel x height x width

        # process label
        label = np.array(Image.open(label_file))
        label = label[:, :, 0]

        return img, label

    def get_inputs(self):
        PSIZE = self.panel_size
        self.inputs = []
        self.labels = []
        self.predictions = []

        for i, j in product(range(0, PSIZE * 2), range(0, PSIZE)):
            image_file, label_file = self.get_files()
            img, label = self.preprocess_images(image_file, label_file)

            # set up input, labels and predictions arrays
            self.inputs.append(img)
            self.labels.append(label)
            self.predictions.append(None)

    def setup_panel(self):
        PSIZE = self.panel_size
        self.get_inputs()

        plt.ion()
        fig, ax = plt.subplots(PSIZE * 2, PSIZE * 2, figsize=(12, 12))

        for i, j in product(range(0, PSIZE * 2), range(0, PSIZE)):
            k = j * PSIZE + i
            label = self.labels[k]
            # do basic scaffolding for the panel
            if (i == 0):
                ax[0, 2 * j].set_title('GT')
                ax[0, 2 * j + 1].set_title('PR')
            ax[i, 2 * j].imshow(label, interpolation='None', cmap='jet', vmin=0, vmax=CaffeLocations.NUM_LABELS)

            ax[i, 2 * j].get_xaxis().set_visible(False)
            ax[i, 2 * j].get_yaxis().set_visible(False)
            ax[i, 2 * j + 1].get_xaxis().set_visible(False)
            ax[i, 2 * j + 1].get_yaxis().set_visible(False)
        plt.draw()
        plt.subplots_adjust(left=0, bottom=0, right=1, wspace=0, hspace=0)

        self.fig = fig
        self.ax = ax

    def show_labels(self):
        PSIZE = self.panel_size
        ax = self.ax
        for i, j in product(range(0, PSIZE * 2), range(0, PSIZE)):
            k = j * PSIZE + i
            label = self.labels[k]
            ax[i, 2 * j].imshow(label, interpolation='None', cmap='jet', vmin=0, vmax=CaffeLocations.NUM_LABELS)
        plt.draw()

    def get_predictions(self):
        """
        Updates the predictions according to the current selected model
        :return:
        """
        PSIZE = self.panel_size
        net = self.net
        for i, j in product(range(0, PSIZE * 2), range(0, PSIZE)):
            k = j + PSIZE * i
            input = self.inputs[k]
            label = self.labels[k]
            net.blobs['data'].data[...] = input
            net.forward()
            pred = net.blobs['score'].data[0]
            # pred = pred[0, ...]
            pred = pred.argmax(axis=0)
            self.predictions[k] = pred

    def get_single_prediction(self, patch_name=None):
        net = self.net
        image_file, label_file = self.get_files(patch_name=patch_name)
        image, label = self.preprocess_images(image_file, label_file)
        net.blobs['data'].data[...] = image
        net.forward()
        pred = net.blobs['score'].data
        pred = pred[0, ...]
        pred = pred.argmax(axis=0)
        return image_file, label, pred

    def show_single_prediction(self, image_file, label, pred):

        # plt.ion()
        fig, ax = plt.subplots(1, 4, figsize=(12, 5))

        fig.canvas.set_window_title(os.path.basename(image_file))

        ax[0].set_title('Input')
        inimage = Image.open(image_file)
        ax[0].imshow(inimage, interpolation='none')

        ax[1].set_title('Centered')
        img1 = (inimage - self.mean).astype(np.uint8)
        ax[1].imshow(img1, interpolation='none')

        ax[2].set_title('GT')
        img2 = ax[2].imshow(label, interpolation='None', cmap='jet', vmin=0, vmax=CaffeLocations.NUM_LABELS)
        ax[2].format_coord = self._get_formatter('Ground Truth', label)

        ax[3].set_title('PR')
        img3 = ax[3].imshow(pred, interpolation='none', vmin=0, vmax=CaffeLocations.NUM_LABELS)
        ax[3].format_coord = self._get_formatter('Prediction', pred)

        for i in range(0, 3):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

        plt.tight_layout()
        # plt.draw()
        plt.show()

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

    def show_predictions(self):
        self.fig.canvas.set_window_title('DeepHisto Sampler [dir: %s, epoch: %d]' % (self.directory, self.epoch))
        PSIZE = self.panel_size
        for i, j in product(range(0, PSIZE * 2), range(0, PSIZE)):
            k = j + PSIZE * i
            pred = self.predictions[k]
            self.ax[i, 2 * j + 1].imshow(pred, interpolation='None', cmap='jet', vmin=0,
                                         vmax=CaffeLocations.NUM_LABELS)
        # plt.tight_layout()
        plt.draw()

    def next_epoch(self, loop=False):
        epoch = self.epoch
        step = self.step
        end = self.end
        if (epoch + step > end):
            if loop:
                self.epoch = self.start
                return True
            else:
                return False

        self.epoch = epoch + step
        self.load_model(self.directory, self.epoch)
        print 'Using epoch [%d]' % epoch
        return True


