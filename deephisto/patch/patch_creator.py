import os
import shutil

import Image
import numpy as np

from deephisto.utils import  Console


class PatchCreator:

    def __init__(self, utils, sampler,target_dir, cleardir=False):
        self.utils = utils
        self.subject = None
        self.index = None
        self.pimages = []
        self.sampler = sampler
        self.target_dir = target_dir

    def clear_dir(self, cleardir):

        data_dir = self.utils.locations.PATCHES_DIR + '/' + self.target_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print 'Creating directory %s' % data_dir
        else:
            #  Empty directory first
            if (cleardir):
                print Console.WARNING + data_dir + ' exists. Emptying the patches directory [%s] before creating new patches' % self.target_dir + Console.ENDC
                shutil.rmtree(data_dir)
                os.makedirs(data_dir)

    def create_patches(self, subject, index ): #,coverage=30):
        """
        :param subject:    The subject
        :param index:       The current slice
        :return:
        """

        self.subject = subject
        self.index   = index
        self.utils.set_subject(subject)

        utils = self.utils

        bmask = self.utils.load_binary_mask_for_slice(index)
        self.sampler.set_mask(bmask)

        self.pimages = []

        images = self.utils.load_sources_for_slice(index)
        for j in images:
            self.pimages.append(Image.fromarray(j))

        histo = self.utils.load_labels_for_slice(index)
        self.pimages.append(Image.fromarray(histo))
        print
        print 'Sampling subject %s, slice %d'%(self.subject, self.index)

        selected = self.sampler.sample()

        count = 0
        for (y,x) in selected:   #first dimension is row (y), second is column (x). results of the sampler in image convention (numpy)
            count = count  + self._extract_patch(x,y) # continue processing in cartesian convention

        print '-----------------------------------------------------------'
        print Console.OKBLUE + Console.BOLD + '%d'%count + Console.ENDC + ' png images were created'
        print '-----------------------------------------------------------'

    def _extract_patch(self, x, y):
        """
        x,y: cartesian coordinates of the center of the patch that needs to be extracted
        """


        filename = self.utils.locations.PATCH_TEMPLATE

        L = len(self.pimages)-1

        samples = []
        WSIDE = self.sampler.WSIDE
        WSIZE = self.sampler.WSIZE

        for i in range(L):
            label = self.utils.locations.LABELS[i]
            image = self.pimages[i]



            cimage = image.crop((x - WSIDE, y -WSIDE, x + WSIDE, y + WSIDE))
            #cimage.save(filename%(self.subject, self.index, label, x, y))  not saving intermediate images
            samples.append(np.array(cimage))

        self.save_multi_channel_image(samples, x, y)

        image = self.pimages[-1]  #  the histology image

        cimage = image.crop((x - WSIDE, y - WSIDE, x + WSIDE, y + WSIDE))
        cimage.save(filename%(self.target_dir, self.subject, self.index, 'HI', x,y))



        return 2 #only count the MU and the HI PATCHES

    def save_multi_channel_image(self, samples, x, y):

        N = len(samples)
        first = samples[0]
        (width, height, _) = first.shape
        multi = np.zeros((width,height,N),'uint8')
        for i in range(N):
            multi[...,i] =  samples[i][...,0]

        img = Image.fromarray(multi)

        filename = self.utils.locations.PATCH_TEMPLATE
        img.save(filename%(self.target_dir, self.subject, self.index,'MU',x,y))






