import pdb
import numpy as np
import matplotlib.pylab as plt

import caffe


class TopoLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) !=2:
            raise Exception('We need two inputs to compute loss')

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):

        label = bottom[1].data[0,...]
        label = label.transpose(1,2,0)
        label = label[:,:,0]

        pred = bottom[0].data[0,...]
        pred = pred.argmax(axis=0)

        diff = np.abs(np.exp(label)-np.exp(pred))
        diff /= diff.max()

        #self._visualize(label, pred, diff)

        self.diff = diff

        top[0].data[...] = np.sum(diff**2)


    def backward(self, top, propagate_down, bottom):
        """
        backward is never called during testing:
        (i.e. caffe test --model=net/dhloss/val.prototxt -- weights=data/dh28s/_iter_300000.caffemodel)
        """
        assert propagate_down[1] != True, 'gradients cannot be calculated with respect to the label inputs'

        if propagate_down[0]:
            bottom_diff = bottom[0].diff

        pdb.set_trace()
        pass


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

        plt.show(block=True)

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