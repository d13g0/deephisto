#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

import os
import shutil

import Image
import numpy as np

from deephisto.utils import Console


class PatchCreator:

    def __init__(self, sampler, utils, config):
        self.sampler = sampler
        self.utils = utils
        self.config = config

        self.subject = None
        self.index = None
        self.labels = None
        self.inputs = []


    def clear_dir(self, cleardir):
        """
        :param cleardir: if True deletes the contents of the patch directory.
        :return:
        """
        PATCH_DIR = self.config.PATCH_DIR

        if not os.path.exists(PATCH_DIR):
            os.makedirs(PATCH_DIR)
            print 'Creating directory %s' % PATCH_DIR
        else:
            #  Empty directory first
            if cleardir:
                print Console.WARNING + PATCH_DIR + ' exists. Emptying the patches directory [%s] before creating new patches' % PATCH_DIR + Console.ENDC
                shutil.rmtree(PATCH_DIR)
                os.makedirs(PATCH_DIR)


    def create_patches(self, subject, index):
        """
        :param subject:    The subject
        :param index:       The current slice
        """
        self.subject = subject
        self.index   = index
        self.utils.set_subject(subject)
        self.inputs = []
        self.labels = None

        bmask = self.utils.load_binary_mask_for_slice(index)
        self.sampler.set_mask(bmask)

        images = self.utils.load_sources_for_slice(index)
        for j in images:
            self.inputs.append(Image.fromarray(j))  # converts the numpy arrays to PIL images. Used for cropping

        self.labels = Image.fromarray(self.utils.load_labels_for_slice(index))

        print
        print 'Sampling subject %s, slice %d'%(self.subject, self.index)

        selected = self.sampler.sample()

        count = 0

        for (y,x) in selected:
            # first dimension is row (y), second is column (x). results of the sampler in numpy convention (numpy)
            count = count  + self.create_patch_files(x, y)  # continue processing in cartesian coords

        print '-----------------------------------------------------------'
        print Console.OKBLUE + Console.BOLD + '%d'%count + Console.ENDC + ' png images were created'
        print '-----------------------------------------------------------'

    def create_patch_files(self, x, y):
        """
        x,y: cartesian coordinates of the center of the patch that needs to be extracted
        """
        WSIDE = self.sampler.WSIDE
        PATCH_TEMPLATE = self.config.PATCH_TEMPLATE

        channels = []
        for i in range(len(self.inputs)):
            image = self.inputs[i]
            input_patch = image.crop((x - WSIDE, y - WSIDE, x + WSIDE, y + WSIDE))
            channels.append(np.array(input_patch))

        self.save_input_patch(channels, x, y)

        filename = PATCH_TEMPLATE.format(subject=self.subject, index=self.index, type='L', x=x, y=y)
        labels_patch = self.labels.crop((x - WSIDE, y - WSIDE, x + WSIDE, y + WSIDE))
        labels_patch.save(filename)
        return 2 # two patches are created

    def save_input_patch(self, channels, x, y):
        """
        Creates the input patch. The order of the channels determines the RGB color
        """

        PATCH_TEMPLATE = self.config.PATCH_TEMPLATE

        N = len(channels)
        first = channels[0]
        (width, height, _) = first.shape
        multi = np.zeros((width, height, N), dtype='uint8')
        for i in range(N):
            multi[...,i] =  channels[i][..., 0]
        img = Image.fromarray(multi)
        filename = PATCH_TEMPLATE.format(subject=self.subject, index=self.index, type='I', x=x, y=y)
        img.save(filename)






