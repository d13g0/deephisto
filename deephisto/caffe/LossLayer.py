import pdb
import itertools
import numpy as np
import matplotlib.pylab as plt
import caffe
import math

from caffe._caffe import layer_type_list


class TopoLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception('We need two inputs to compute loss')
        self.counter = 0

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):

        pred = np.argmax(bottom[0].data[0, ...], axis=0)
        label = bottom[1].data[0, ...]
        label = label.transpose(1, 2, 0)
        label = label[:, :, 0]

        inputs = bottom[0].data[0, ...]
        scores = inputs - np.max(inputs, axis=0)
        exp_scores = np.exp(scores)

        if np.all(np.sum(exp_scores, axis=0, keepdims=True)) == 0:
            print 'PROBLEM'
            pdb.set_trace()

        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        if not np.amax(probs) <= 1 or not np.amin(probs) >= 0:
            print 'PROBLEM'
            pdb.set_trace()

        expected_value = np.zeros_like(pred, dtype=np.float32)
        for i in range(0, len(probs)):
            k = i
            expected_value += probs[i] * k

        diff = np.exp(label) - np.exp(expected_value)

        loss = 0
        _, h, w = probs.shape
        for i in range(h):
            for j in range(w):
                idx = int(label[i, j])
                loss += log_prob[idx, i, j]  # loss only sum the log_probs corresponding to the true classes

        self.counter += 1

        if self.counter == 100:
            self._visualize(label, pred, diff)
            titles = ['1. input', '2. scores', '3. exp_scores', '4. probability', '5.log prob']
            steps = [inputs, scores, exp_scores, probs, log_prob]
            self._show_channels(steps, titles)
            plt.show()
            self.counter = 0

        self.probs = probs
        self.loss = loss
        self.label = label
        top[0].data[...] = self.loss

    def backward(self, top, propagate_down, bottom):
        """
        backward is never called during testing:
        (i.e. caffe test --model=net/dhloss/val.prototxt -- weights=data/dh28s/_iter_300000.caffemodel)
        """
        assert propagate_down[1] != True, 'gradients cannot be calculated with respect to the label inputs'

        loss = self.loss
        grad = self.probs
        label = self.label

        _, h, w = grad.shape
        for i in range(h):
            for j in range(w):
                idx = int(label[i, j])
                grad[idx, i, j] = (grad[idx, i, j] - 1) / loss

        # assert np.any(np.max(np.abs(self.grad))) <=  math.exp(10), 'numeric problem'

        print 'loss %.2f  min: %f max: %f' % (self.loss, grad.min(), grad.max())

        if propagate_down[0]:
            bottom[0].diff[0, ...] = grad

    def _visualize(self, label, pred, diff):

        fig, ax = plt.subplots(1, 3, squeeze=True)

        fig.canvas.set_window_title('Prediction')

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

    def _show(self, title, image, use_label_range=True):
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        if use_label_range:
            plt.imshow(image, interpolation='none', vmin=0, vmax=10, cmap='jet')
        else:
            plt.imshow(image, interpolation='none', cmap='jet')
        ax = plt.gca()
        ax.format_coord = self._get_formatter(title, image)

    def _show_channels(self, steps, titles):

        N = len(steps)
        C = len(steps[0])

        fig, ax = plt.subplots(ncols=C, nrows=N, squeeze=False, figsize=(16, 8), facecolor='black')

        fig.canvas.set_window_title('Loss Layer Processing')

        for i, step in enumerate(steps):
            ax[i, 0].set_ylabel(titles[i], rotation=90, size='large', color='#cccccc')
            for j in range(C):
                ax[i, j].set_axis_bgcolor('black')
                ax[i, j].imshow(step[j], interpolation='none', vmin=step[j].min(), vmax=step[j].max(), cmap='hot')
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_ticks([])
                ax[i, j].format_coord = self._get_formatter('%s - Channel [%d]' % (titles[i], j), step[j])

        plt.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.05, wspace=0.0, hspace=0.0)

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
