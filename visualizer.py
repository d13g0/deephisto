# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:35:15 2016

@author: dcantor
"""

import matplotlib as mpl
import numpy as np

mpl.use('TkAgg', warn=False)
import matplotlib.pylab as plt
import matplotlib.patches as ppa
from matplotlib.widgets import Button
from matplotlib import gridspec
import Image
import Tkinter as Tk

from deephisto.locations import Locations
from deephisto.image import ImageUtils
from deephisto.patch import PatchSampler
from deephisto.utils import DraggableRectangle



class SliceButtonCallback(object):
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def next(self, event):
        if not self.visualizer.next_slice():
            print 'LAST SLICE REACHED'

    def prev(self, event):
        if not self.visualizer.previous_slice():
            print 'FIRST SLICE REACHED'


class ImageSetHelper:
    def load(self, utils, indices):

        self.mask_set = []
        self.image_set = []
        self.histo_set = []  # contains the real intensities
        self.pimage_set = []  # used for cropping with PIL

        print 'Loading Image Set'
        for i in indices:
            print 'Slice [%d]' % i
            mask = utils.load_mask_png(i)
            images = utils.load_source_png_images(i)
            histo = utils.load_histo_png_image(i)
            pimages = []

            self.mask_set.append(mask)
            self.image_set.append(images)
            self.histo_set.append(histo)

            for j in images:
                pimages.append(Image.fromarray(j))
            pimages.append(Image.fromarray(histo))

            self.pimage_set.append(pimages)

        print 'Image Set Loaded'


class GoToDialog(object):
    def __init__(self, visualizer):

        self.visualizer = visualizer
        self.master = None

    def show(self):

        if self.master is not None:
            self.master.destroy()

        master = self.master = Tk.Tk()
        master.wm_title('Go to:')
        Tk.Label(self.master, text='x,y :').grid(row=1)
        e1 = self.e1 = Tk.Entry(master, width=10)
        e2 = self.e2 = Tk.Entry(master, width=10)
        e1.grid(row=1, column=1, padx=1, pady=2)
        e2.grid(row=1, column=2, padx=1, pady=2)
        Tk.Button(master, text='Go', command=self.goto_coords).grid(row=3, column=1, sticky=Tk.S, pady=2)
        Tk.Button(master, text='Cancel', command=self.quit).grid(row=3, column=2, sticky=Tk.S, pady=2)
        e1.focus_set()
        e1.focus_force()
        self.master.pack_propagate()
        self.center_window()
        Tk.mainloop()

    def goto_coords(self):
        # Using cartesian coordinates
        x = int(self.e1.get())
        y = int(self.e2.get())
        self.visualizer.update_patch(x, y)
        self.master.destroy()
        self.master = None

    def quit(self):
        self.master.destroy()
        self.master = None

    def center_window(self, width=270, height=150):
        mng = self.visualizer.fig.canvas.manager
        print mng.window
        if mng.window is not None and isinstance(mng.window, Tk.Tk):
            x, y = mng.window.winfo_pointerxy()
        else:
            master = self.master
            screen_width = master.winfo_screenwidth()
            screen_height = master.winfo_screenheight()
            x = (screen_width / 2) - (width / 2)
            y = (screen_height / 2) - (height / 2)
        self.master.geometry('%dx%d+%d+%d' % (width, height, x, y))


class Visualizer:
    """
    Visualizes the DeepHisto database. Coordinates appear in cartesian convention,
    with the origin in the upper left corner:

              cols
        ----------------> x (w)
        |
  rows  |
        |
        v y (h)
    """

    def __init__(self, locations):

        print ' Patch Visualizer'
        print '---------------------------------------\n'
        self.fig = None
        self.pimages = []
        self.drs = []
        self.axs = []
        self.bxs = []
        self.vmin = 0
        self.vmax = 255
        self.utils = ImageUtils(locations)
        self.image_helper = ImageSetHelper()
        self._go_dialog = GoToDialog(self)
        self._picker = None
        if locations.subject is not None:
            self.set_subject(locations.subject)

    def reset(self):

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        self.pimages = []
        self.drs = []
        self.axs = []
        self.bxs = []
        self.vmin = 0
        self.vmax = 255

    def set_subject(self, subject):

        self.reset()

        print 'Setting subject for PatchVisualizer'
        print '---------------------------------------'

        self.subject = subject

        print 'Subject [%s]' % self.subject
        self.utils.set_subject(subject)

        (self.vmin, self.vmax) = self.utils.get_dynrange_histo()  # makes sure the range is uniform for all slices
        print 'Dynamic range for histology: %d, %d' % (self.vmin, self.vmax)

        self.indices = self.utils.get_annotation_indices()
        print 'Available slices: %s' % self.indices

        self.image_helper.load(self.utils, self.indices)

        slice_num = self.indices[0]
        print 'Selecting first slice: [%s]' % slice_num
        self.set_slice(slice_num)

    def set_slice(self, i):

        self.slice_num = i

        print 'Setting slice [%d]' % i
        idx = self.indices.index(i)
        self.mask = self.image_helper.mask_set[idx]
        self.images = self.image_helper.image_set[idx]
        self.histo = self.image_helper.histo_set[idx]
        self.histo_flat = self.histo[:, :, 0]  # to show with color maps
        self.pimages = self.image_helper.pimage_set[idx]
        self.num_sources = len(self.pimages)

    def next_slice(self):
        indices = self.indices
        idx = indices.index(self.slice_num)
        if (idx < len(indices) - 1):
            self.set_slice(indices[idx + 1])
            self.update()
            return True
        else:
            return False

    def previous_slice(self):
        indices = self.indices
        idx = indices.index(self.slice_num)
        if (idx > 0):
            self.set_slice(indices[idx - 1])
            self.update()
            return True
        else:
            return False

    def get_annotation_indices(self):
        return self.utils.get_annotation_indices()

    def init(self):
        """
        Initializes the main visualization
        """

        images = self.images
        mask = self.mask

        L = self.num_sources - 1  # all sources but histo image (which is the last in the images list

        # plt.ion()
        self.fig = plt.figure(facecolor='black')
        self.fig.canvas.set_window_title(
                'DeepHisto subject: %s slice:%s patch: (%d, %d)' % (self.subject, self.slice_num, 0, 0))
        self.fig.suptitle('DeepHisto subject: %s slice:%s patch: (%d, %d)' % (self.subject, self.slice_num, 0, 0),
                          fontsize=18, color='white')

        mpl.rcParams['keymap.save'] = ''

        def keyevent_handler(event):
            if event.key == 's':
                print 'Saving figure /home/dcantor/Desktop/%s_%s.png' % (self.subject, self.slice_num)
                event.canvas.figure.savefig('/home/dcantor/Desktop/%s_%s.png' % (self.subject, self.slice_num),
                                            facecolor=self.fig.get_facecolor(), edgecolor='none')

            if event.key == 'g':
                self._go_dialog.show()

        self.fig.canvas.mpl_connect('key_press_event', keyevent_handler)

        gs = gridspec.GridSpec(L * 2, 8)

        first_ax = None

        for i in range(L):
            A = i * 2
            B = A + 2
            if (i == 0):
                ax = self.fig.add_subplot(gs[A:B, 0:2])
                first_ax = ax
            else:
                ax = self.fig.add_subplot(gs[A:B, 0:2], sharex=first_ax, sharey=first_ax)

            ax.set_axis_bgcolor('black')
            ax.imshow(images[i], interpolation='none')
            ax.imshow(mask, alpha=0.3)
            ax.get_xaxis().set_visible(False)
            ax.set_ylabel(self.utils.locations.LABELS[i], fontsize=16, color='#cccccc', rotation=0)
            ax.get_yaxis().set_ticks([])

            bx = self.fig.add_subplot(gs[A:B, 2:4])
            bx.get_xaxis().set_visible(False)
            bx.get_yaxis().set_visible(False)

            self.axs.append(ax)
            self.bxs.append(bx)

        # Now processing histo image

        ax = self.fig.add_subplot(gs[0:2, 4:6], sharex=first_ax, sharey=first_ax)
        ax.set_axis_bgcolor('black')
        self._histo_flat_im = ax.imshow(self.histo_flat, interpolation='none', vmin=self.vmin, vmax=self.vmax,
                                        cmap='jet')

        ax.imshow(mask, alpha=0.3)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Histology', fontsize=16, color='#cccccc')
        ax.get_xaxis().set_ticks([])

        bx = self.fig.add_subplot(gs[0:2, 6:8])
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)

        self.axs.append(ax)
        self.bxs.append(bx)

        callback = SliceButtonCallback(self)

        bprev_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(bprev_ax, 'Previous')
        self.bprev.on_clicked(callback.prev)

        bnext_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.bnext = Button(bnext_ax, 'Next')
        self.bnext.on_clicked(callback.next)

        mng = plt.get_current_fig_manager()
        if mng.window is not None and isinstance(mng.window, Tk.Tk):
            mng.window.attributes('-zoomed', True)

        self.fig.show()

    def _zoom_source(self, x, y):
        """
        Zooms in the source images
        x,y: cartesian coordinates of the center of the patch
        """
        L = self.num_sources - 1  # all sources but histo image (which is the last in the images list

        for i in range(L):
            pimage = self.pimages[i]
            cimage = pimage.crop(
                    (x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
            self.bxs[i].imshow(cimage, interpolation='none')

    def _zoom_histo(self, x, y):
        """
        Zooms in the histology
        x,y: cartesian coordinates of the center of the patch
        """

        im = Image.fromarray(self.histo).crop(
                (x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
        im = np.array(im)[:, :, 0]
        self._imhisto = self.bxs[-1].imshow(im, interpolation='none', vmin=self.vmin, vmax=self.vmax, cmap='jet')
        self._update_pixel_picker(im)

    def _update_pixel_picker(self, zoomed_histo):

        numcols, numrows = zoomed_histo.shape

        def format_coord(x, y):
            row = int(x + 0.5)
            col = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = zoomed_histo[col, row]
                return 'x=%d, y=%d, Neuronal Count =%d' % (col, row, z)
            else:
                return 'x=%d, y=%d' % (y, x)

        self.bxs[-1].format_coord = format_coord

    def _create_draggable_rectangles(self):
        """
        Creates the draggable rectangles
        :return:
        """
        x = self._patch_x
        y = self._patch_y
        for i, ax in enumerate(self.axs):
            self._ax = self.axs[i]
            rect = ppa.Rectangle((x - PatchSampler.WSIDE, y - PatchSampler.WSIDE), PatchSampler.WSIZE,
                                 PatchSampler.WSIZE, linewidth=1, edgecolor='r', facecolor='none')
            self._ax.add_patch(rect)
            dr = DraggableRectangle(rect, self.update_patch)
            dr.connect()
            self.drs.append(dr)

    def _update_title(self):
        """
        Updates the title
        :return:
        """
        if (self.fig is not None):
            self.fig.canvas.set_window_title('DeepHisto subject: %s slice:%s patch: (%d, %d)' % (
                self.subject, self.slice_num, self._patch_x, self._patch_y))
            self.fig.suptitle('DeepHisto subject: %s slice:%s patch: (%d, %d)' % (
                self.subject, self.slice_num, self._patch_x, self._patch_y), fontsize=18, color='white')

    def create_patch(self, x, y):
        '''
        Creates interactive patches (draggable)
        This method is invoked by the user (ideally once)
        x,y: cartesian coordinates of the center of the wanted patch
        '''
        self._patch_x = x  # coordinates of the patch centre
        self._patch_y = y
        self._create_draggable_rectangles()
        self._update_title()
        self._zoom_source(x, y)
        self._zoom_histo(x, y)

        im = Image.fromarray(self.histo).crop(
                (x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
        im = np.array(im)[:, :, 0]
        numrows, numcols = im.shape

        self._update_pixel_picker(im)
        #@TODO: show a color bar
        # clbar = plt.colorbar(self._histo_flat_im , ax=self.bxs[-1], ticks=range(self.vmin, self.vmax + 1, 2))
        # plt.setp(plt.getp(clbar.ax.axes, 'yticklabels'), color='w')

        plt.show()

    def update_patch(self, x, y):
        """
        Updates the position of the current patch
        x,y: cartesian coordinates of the center of the patch
        """
        self._patch_x = x
        self._patch_y = y
        self._update_title()
        self._zoom_source(x, y)
        self._zoom_histo(x, y)
        for i, ax in enumerate(self.axs):
            self._ax = ax
            self.drs[i].rect.set_x(x - PatchSampler.WSIDE)
            self.drs[i].rect.set_y(y - PatchSampler.WSIDE)
        self.fig.canvas.draw()

    def update(self):
        """
        Updates the current slice (on a multislice patient)
        :return:
        """
        x = self._patch_x
        y = self._patch_y
        self._update_title()
        images = self.images
        mask = self.mask
        for i, ax in enumerate(self.axs[:-1]):
            ax.imshow(images[i], interpolation='none')
            ax.imshow(mask, alpha=0.3)
        self.axs[-1].imshow(self.histo_flat, interpolation='none', vmin=self.vmin, vmax=self.vmax)
        self.axs[-1].imshow(mask, alpha=0.3)
        self.update_patch(self._patch_x, self._patch_y)
