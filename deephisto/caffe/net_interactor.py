import os

os.environ['GLOG_minloglevel'] = '2'

import os.path
import sys, pdb
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import caffe
from PIL import Image
from CaffeLocations import CaffeLocations


# Making sure that this file is visible to caffe
sys.path.insert(0, CaffeLocations.CAFFE_CODE_DIR)

class NetInteractor:
    def __init__(self):

        caffe.set_mode_gpu()
        self.filenames = None
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
        self.validation_data = None
        self.training_data = None
        self.mean = CaffeLocations.TRAINING_MEAN #None
        self.data_dir = None
        self.PANEL_ROWS = 5
        self.PANEL_COLS = 4
        self.verbose = True

    def load_avg_image(self):
        avg_img_file = CaffeLocations.SPLIT_DIR + '/' + self.data_dir + '/' + CaffeLocations.AVG_IMG
        self.mean = np.array(Image.open(avg_img_file)) #CaffeLocations.TRAINING_MEAN #
        #print 'Average training image %s loaded'%avg_img_file


    def load_lists(self, data_dir=None):

        if data_dir == None:
            data_dir = self.data_dir

        datafile = CaffeLocations.SPLIT_DIR + '/%s/validation.txt'%data_dir
        with open(datafile, 'r') as dfile:
            lines = dfile.read().splitlines()
            self.validation_data = [f.split(';')[0] for f in lines]
            #print 'Training split %s loaded' % datafile

        datafile = CaffeLocations.SPLIT_DIR + '/%s/training.txt'%data_dir
        with open(datafile, 'r') as dfile:
            lines = dfile.read().splitlines()
            self.training_data = [f.split(';')[0] for f in lines]
            #print 'Validation split %s loaded'%datafile

        return self.training_data, self.validation_data

    def set_animation_params(self, directory, start, end, step, data_dir):
        self.directory = directory
        self.start = start
        self.end = end
        self.step = step
        self.load_model(directory, start, data_dir)

        if self.verbose:

            print
            print 'Animation params'
            print '-----------------'
            print '  net dir         :%s'%directory
            print '  data dir        :%s' % data_dir
            print '  start           :%d'%start
            print '  end             :%d'%end
            print '  step            :%d'%step


    def load_model(self, directory, epoch, data_dir):
        """
        :param directory: The directory where the caffe trained .caffemodels exist
        :param epoch:  the index X of the file __iter__X__.caffemodel to be used
        :param data_dir: location under SPLIT_DIR and PATCH_DIR to look for data (must be the same name)
        """
        self.directory = directory
        self.epoch = epoch
        self.data_dir = data_dir

        #loads the average image annd the validation and training lists from the
        #respective subdirectory under CaffeLocations.SPLIT_DIR
        self.load_avg_image()
        self.load_lists()

        #loads the  caffe network (deploy.prototxt) with the respective weights
        #ready to make predictions
        weights = CaffeLocations.SNAPSHOT_DIR % (directory, epoch)
        model = CaffeLocations.NET_DIR + '/'+ directory +'/'+ CaffeLocations.DEPLOY_PROTO
        self.net = caffe.Net(model, caffe.TEST, weights=weights)

        if self.verbose:
            print 'directory: %s epoch: %s  data:%s' % (directory, epoch, data_dir)

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
        """
        Looks for files under CaffeLocations.PATCHES_DIR/self.data_dir
        """

        if patch_name is None:
            # if not specific patch is sought. Then get a random one from the validation list
            N = len(self.validation_data)
            idx = np.random.randint(0, N - 1)
            image_file = CaffeLocations.PATCHES_DIR + '/'+ self.data_dir +'/'+ self.validation_data[idx]
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

            image_file = CaffeLocations.PATCHES_DIR + '/'+ self.data_dir +'/' + image_file
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
        img = img.transpose((2, 0, 1))  # transpose to channel x height x width

        # process label
        label = np.array(Image.open(label_file))

        label = label[:, :, 0]

        # label[(label < 1)] = 0
        # label[(label >= 1)] = 5  # surgery

        return img, label

    def get_inputs(self):

        NUM_INPUTS = self.PANEL_COLS * self.PANEL_ROWS

        self.inputs = []
        self.labels = []
        self.filenames = []

        for k in range(NUM_INPUTS):
            image_file, label_file = self.get_files()
            self.filenames.append(image_file)
            img, label = self.preprocess_images(image_file, label_file)
            self.inputs.append(img)
            self.labels.append(label)

    def get_predictions(self):
        """
        Updates the predictions according to the current selected model
        :return:
        """
        NUM_INPUTS = self.PANEL_COLS * self.PANEL_ROWS
        net = self.net
        self.predictions = []

        for k in range(NUM_INPUTS):
            input = self.inputs[k]
            label = self.labels[k]
            net.blobs['data'].data[...] = input
            net.forward()
            pred = net.blobs['score'].data
            pred = pred[0, ...]
            pred = pred.argmax(axis=0)
            self.predictions.append(pred)

    def setup_panel(self):
        ROWS = self.PANEL_ROWS
        COLS = self.PANEL_COLS

        fig, ax = plt.subplots(ROWS, COLS * 2,  squeeze=False, facecolor='black', figsize=(12,8))
        fig.canvas.set_window_title('DeepHisto Training [dir: %s, epoch: %d]' % (self.directory, self.epoch))
        fig.suptitle('DeepHisto Training [dir: %s, epoch: %d]' % (self.directory, self.epoch))

        for i, j in product(range(0,  ROWS), range(0, COLS*2)):
            k = j*ROWS + i

            if (i == 0):
                if (j % 2 ==0):
                    ax[i, j].set_title('GT', color='white')
                else:
                    ax[i,j].set_title('PR', color='white')

            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].set_axis_bgcolor('black')
        plt.subplots_adjust(top=0.9, left=0, bottom=0, right=1, wspace=0, hspace=0)
        plt.draw()
        self.fig = fig
        self.ax = ax

    def show_labels(self):

        ROWS = self.PANEL_ROWS
        COLS = self.PANEL_COLS
        N = ROWS * COLS

        assert len(self.labels) == N, "number of labels do not correspond to the dimension of the panel"

        ax = self.ax
        for i, j in product(range(0, ROWS), range(0, COLS)):
            k = j * ROWS + i
            label = self.labels[k]
            ax[i,2*j].imshow(label, interpolation='None', cmap='jet', vmin=0, vmax=CaffeLocations.NUM_LABELS)
            ax[i,2*j].format_coord = self._get_formatter('Source:  %s' % self.filenames[k], label)
        plt.subplots_adjust(top=0.9, left=0, bottom=0, right=1, wspace=0, hspace=0)
        plt.draw()

    def show_predictions(self):

        ROWS = self.PANEL_ROWS
        COLS = self.PANEL_COLS
        N = ROWS * COLS

        assert len(self.predictions) == N, 'number of predictions need to match the dimensions of the panel'

        self.fig.canvas.set_window_title('DeepHisto Training [dir: %s, epoch: %d]' % (self.directory, self.epoch))
        self.fig.suptitle('%d' % self.epoch, fontsize=22, color='white')

        ax = self.ax
        for i, j in product(range(0, ROWS), range(0, COLS)):
            k = j * ROWS + i
            pred = self.predictions[k]
            self.ax[i, 2 * j + 1].imshow(pred, interpolation='None', cmap='jet', vmin=0,
                                         vmax=CaffeLocations.NUM_LABELS)
            self.ax[i, 2 * j + 1].format_coord = self._get_formatter('Prediction: value %d', pred)
        plt.subplots_adjust(top=0.9, left=0, bottom=0, right=1, wspace=0, hspace=0)
        plt.draw()

    def get_single_prediction(self, patch_name=None):
        net = self.net
        image_file, label_file = self.get_files(patch_name=patch_name)
        image, label = self.preprocess_images(image_file, label_file)
        net.blobs['data'].data[...] = image
        net.forward()
        pred = net.blobs['score'].data
        pred = pred[0, ...]

        N,_,_ = pred.shape
        channels = []
        for i in range(N):
            channels.append(pred[i])
        pred = pred.argmax(axis=0)
        return image_file, label, pred, channels

    def show_single_prediction(self, image_file, label, pred, channels):


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

        for i in range(0, 4):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.draw()

        N = len(channels)
        fig, ax = plt.subplots(1, N, figsize=(16,3))
        fig.canvas.set_window_title('Channels')
        for i in range(N):
            ax[i].set_title('%d'%i)
            ax[i].imshow(channels[i], interpolation='none', vmin=0,vmax=CaffeLocations.NUM_LABELS)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].format_coord = self._get_formatter('Channel %d'%i, channels[i])

        plt.tight_layout()
        plt.draw()

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
        self.load_model(self.directory, self.epoch, self.data_dir)
        return True


# if __name__=='__main__':
#   FOR DEBUG
#   inter = NetInteractor()
#   inter.load_model('dh28',40000,'28x28')
#   inter.setup_panel()
#   inter.get_inputs()
#   inter.get_predictions()
#   inter.show_labels()
#   inter.show_predictions()
#   plt.show()
