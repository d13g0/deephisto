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

        self.pred = np.argmax(bottom[0].data[0, ...], axis=0)
        self.label = bottom[1].data[0, ...]
        self.label = self.label.transpose(1, 2, 0)
        self.label = self.label[:, :, 0]

        inputs = bottom[0].data[0, ...]
        self.inputs = inputs - np.max(inputs, axis=0)
        exp_scores = np.exp(inputs)


        assert np.any(np.sum(exp_scores, axis=0, keepdims=True)>0), 'denominator cannot be zero or negative'

        self.probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        assert self.probs.min() >=0, 'probability cannot be negative'

        log_probs = -np.log(np.maximum(self.probs, np.finfo('f').min))

        self.Tmaps = self._get_T(self.label, self.probs)
        self.Gmaps = self._get_G(self.probs)
        log_gmaps = -np.log(np.maximum(self.Gmaps, 1e-10))
        log_imgmaps = -np.log(np.maximum(1-self.Gmaps, 1e-10))
        self.diff = np.sum(log_gmaps * self.Tmaps + log_imgmaps * (1 - self.Tmaps), axis=0)
        self.loss = np.sum(self.diff)

        # self.expected_value = np.zeros_like(self.pred, dtype=np.float32)
        # for i in range(len(self.probs)):
        #     k = i
        #     self.expected_value += self.probs[i] * k
        #
        # self.base = self.label - self.expected_value
        # self.diff = self.base ** 2 /100
        #
        # _, h, w = self.probs.shape
        #
        # log_loss = 0
        # for i in range(h):
        #     for j in range(w):
        #         idx = int(label[i, j])
        #         if (idx >= 10):
        #             idx = 9
        #         log_loss += log_probs[idx, i, j]  # loss only sum the log_probs corresponding to the true classes
        #
        # self.alpha = 0.6
        #
        # self.loss = log_loss + np.sum(self.diff)#self.alpha * (log_loss) + (1 - self.alpha) * np.sum(self.diff)


        #smoothness term
        # nmaps = self._get_Nmaps(self.probs, label)
        # sum_nmaps = np.sum(nmaps, axis=0)
        # self._visualize(self.label, self.pred, self.diff)
        # self._show_channels([inputs, self.probs, nmaps],['1.input','2.probs','3.nmaps'])
        # self._show(sum_nmaps,'sum_nmaps')
        # plt.show()

        top[0].data[...] = self.loss

    def backward(self, top, propagate_down, bottom):
        """
        backward is never called during testing:
        (i.e. caffe test --model=net/dhloss/val.prototxt -- weights=data/dh28s/_iter_300000.caffemodel)
        """
        assert propagate_down[1] != True, 'gradients cannot be calculated with respect to the label inputs'

        if propagate_down[0]:

            probs = self.probs
            loss = self.loss
            label = self.label
            pred = self.pred
            diff = self.diff
            inputs = self.inputs

            grads_1 = self.probs
            #expected_value = self.expected_value

            dL_dG = - self.Tmaps / (self.Gmaps+1e-10) + (1 - self.Tmaps)/(1-self.Gmaps+1e-10)
            dL_dP = self._get_dL_dP(dL_dG, probs)
            dL_dI = self._get_dL_dI(dL_dP, probs)

            bottom[0].diff[0,...] = dL_dI


            # N, h, w = grads_1.shape
            # for i in range(h):
            #     for j in range(w):
            #         idx = int(label[i, j])
            #         if idx >= 10:
            #             idx = 9
            #         grads_1[idx, i, j] -= 1
            #
            # sum = 0
            #
            # dP_dI = self._get_dP_dI(probs)
            # grads_2 = np.zeros_like(probs)
            # for i in range(N):
            #     grads_2[i] = - 2 * self.base * dP_dI[i]  / 100
            #
            # grads = grads_1 + grads_2
            #
            # bottom[0].diff[0, ...] = grads

            # print 'loss %.2f  min %f, max %f' % (self.loss, dL_dI.min(), dL_dI.max())
            # self.counter += 1
            # if self.counter == 1000 :
            #     imdiff = self.diff
            #     self._visualize(label, pred, imdiff)
            #     self._show_channels([inputs, probs, self.Tmaps,self.Gmaps,dL_dG,dL_dP,dL_dI],['1.input','2.probs','Tmaps','Gmaps','3.dL_dG','4.dL_dP','5.dL_dI'])
            #     plt.show()
            #     self.counter = 0

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

    def _show(self, image, title, use_label_range=True):
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

                return '%s x,y: (%d, %d)  value: %.5f' % (title, col, row, z)
            else:
                return '%s x,y: (%d, %d)' % (title, col, row)

        return formatter

    def _get_G(self, probs):
        N, _, _ = probs.shape
        Gmap = np.zeros_like(probs)
        Gmap[0] = probs[0]

        for i in range(1, N):
            for j in range(i, N):
                Gmap[i] = Gmap[i] + probs[j]

        return Gmap

    def _get_T(self, label, probs):
        Tmap = np.zeros_like(probs)
        N, _, _ = probs.shape

        Tmap[0] = np.array(label == 0, dtype=np.int)

        for i in range(1, N):
            Tmap[i] = np.array(label >= i, dtype=np.int)

        return Tmap

    def _get_dL_dP(self, dL_dG, probs):
        dL_dP = np.zeros_like(probs)
        N, _, _ = probs.shape

        for i in range(N):
            for j in range(N):
                if (i == j == 0) or (i >= j and j >= 1):
                    dL_dP[i] += dL_dG[j]
        return dL_dP

    def _get_dL_dI(self, dL_dP, probs):
        dL_dI = np.zeros_like(probs)
        N, _, _ = probs.shape

        for i in range(N):
            for j in range(N):
                if (i == j):
                    dL_dI[i] += dL_dP[j] * probs[i] * (1 - probs[i])
                else:
                    dL_dI[i] -= dL_dP[j] * probs[i] * probs[j]
        return dL_dI

    def _get_dP_dI(self, probs):
        dP_dI = np.zeros_like(probs)
        N, _, _ = probs.shape

        for i in range(N):
            for j in range(N):
                if (i == j):
                    dP_dI[i] += j * probs[j] * (1 - probs[j])
                else:
                    dP_dI[i] -= j * probs[j] * probs[i]
        return dP_dI




    def _get_Nmaps(self, probs, label):

        def ev_neighbour(gt, i,j,k,l):
            lbl = int(label[i,j])
            if lbl >= 10:
                lbl = 9
            if label[i,j] == label[k,l] and lbl == gt:
                return math.fabs(probs[lbl,i,j] - probs[lbl,k,l])
            else:
                return 0

        N, h, w = probs.shape

        Nmaps = np.zeros_like(probs)
        for gt in range(N):
            for i in range(h):
                for j in range(w):
                    suma = 0
                    if i-1>=0:
                        suma += ev_neighbour(gt, i,j,i-1,j)
                    if i+1<h:
                        suma += ev_neighbour(gt,i,j,i+1,j)
                    if j-1>=0:
                        suma += ev_neighbour(gt,i,j,i,j-1)
                    if j+1<w:
                        suma += ev_neighbour(gt, i,j,i,j+1)

                    Nmaps[gt, i, j] = suma

        return Nmaps
