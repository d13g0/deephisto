import pdb
import numpy as np
import matplotlib.pylab as plt
import caffe

from caffe._caffe import layer_type_list


class TopoLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) !=2:
            raise Exception('We need two inputs to compute loss')
        self.counter = 0

    def reshape(self, bottom, top):

        self.height = bottom[0].height
        self.width = bottom[0].width
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):

        label = bottom[1].data[0,...]
        label = label.transpose(1,2,0)
        label = label[:,:,0]

        pred = bottom[0].data[0,...]
        pred = pred.argmax(axis=0)

        scores = bottom[0].data
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores,axis=1, keepdims=True)
        probs = probs[0,...]  #remove batch dim

        max_prob = np.argmax(probs, axis=0)


        expected_value = np.zeros_like(pred, dtype=np.float32)
        for i in range(0,10):
            k = i+1 #shift
            expected_value += probs[i] * k

        expected_value -= 1 #shifting back


        diff  = np.abs(np.exp(label)-np.exp(pred))
        diff2 = np.abs(np.exp(label)-np.exp(expected_value))


        self.counter += 1

        if self.counter == 10:

            self._visualize(label, pred, diff)
            self._show_channels(bottom[0].data[0,...], title='input channels')

            self._show_channels(probs, title='probablity maps after soft-max')
            self._show('max_prob', max_prob)
            self._show('expected', expected_value)
            self._show('loss with expected value (now differentiable)',diff2, use_label_range=False)

            plt.show()
            pdb.set_trace()

        #self.diff = diff
        self.loss = np.sum(diff2**2)

        top[0].data[...] = self.loss


    def backward(self, top, propagate_down, bottom):
        """
        backward is never called during testing:
        (i.e. caffe test --model=net/dhloss/val.prototxt -- weights=data/dh28s/_iter_300000.caffemodel)
        """
        assert propagate_down[1] != True, 'gradients cannot be calculated with respect to the label inputs'

        if propagate_down[0]:

            bottom_diff = bottom[0].diff[0,...]

            print 'loss %.2f count: %d, min: %.2f max: %.2f'%(self.loss, self.counter, bottom_diff.min(), bottom_diff.max())

            label = bottom[1].data[0, ...]
            label = label.transpose(1, 2, 0)
            label = label[:, :, 0]

            pred = bottom[0].data[0, ...]
            pred = pred.argmax(axis=0)

            for i in range(0, self.height):
                for j in range(0, self.width):
                    bottom_diff[int(label[i,j]),i,j] = self.diff[i,j]


            if self.counter == 10:
                self._show_channels(bottom_diff, title='backward diff')
                plt.show()
                self.counter = 0



    def _visualize(self, label, pred, diff):

        fig, ax = plt.subplots(1, 3, squeeze=True)

        for i in range(0, 3):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

        ax[0].set_title('label')
        ax[0].imshow(label, interpolation='none', vmin=0, vmax=10, cmap='jet')
        ax[0].format_coord = self._get_formatter('label', label)

        ax[1].set_title('prediction')
        ax[1].imshow(pred, interpolation='none', vmin=0, vmax=10, cmap='jet')
        ax[1].format_coord = self._get_formatter('prediction', pred)

        ax[2].set_title('diff')
        axim = ax[2].imshow(diff, interpolation='none')
        ax[2].format_coord = self._get_formatter('difference', diff)

        plt.tight_layout()

    def _show(self, title, image, use_label_range = True):
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        if use_label_range:
            plt.imshow(image, interpolation='none', vmin=0, vmax=10, cmap='jet')
        else:
            plt.imshow(image, interpolation='none', cmap='jet')
        ax = plt.gca()
        ax.format_coord = self._get_formatter(title, image)

    def _show_channels(self, channels, title=None):
        N = len(channels)

        fig, ax = plt.subplots(1, N, figsize=(16, 3))
        fig.canvas.set_window_title('Channels %s'%title)
        for i in range(N):
            ax[i].set_title('%d' % i)
            ax[i].imshow(channels[i], interpolation='none', vmin = channels.min(), vmax= channels.max(), cmap='gist_heat')
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].format_coord = self._get_formatter('Channel %d' % i, channels[i])

        plt.tight_layout()


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