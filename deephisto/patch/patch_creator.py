import os
import shutil

import Image
import numpy as np

from deephisto.patch import PatchSampler
from deephisto.utils import  Console


class PatchCreator:

    def __init__(self, utils):
        self.utils = utils
        self.subject = None
        self.index = None
        self.pimages = []

    def create_patches(self, subject, index, cleardir=False): #,coverage=30):
        """
        :param subject:    The subject
        :param index:       The current slice
        :return:
        """

        self.subject = subject
        self.index   = index
        self.utils.set_subject(subject)

        utils = self.utils
        if not os.path.exists(utils.locations.PATCHES_DIR):
            os.makedirs(utils.locations.PATCHES_DIR)
            print 'Creating directory %s'%utils.locations.PATCHES_DIR
        else:
            #  Empty directory first
            if (cleardir):
                print Console.WARNING + utils.locations.PATCHES_DIR + ' exists. Emptying the patches directory [%s] before creating new patches'%utils.locations.PATCHES_DIR + Console.ENDC
                shutil.rmtree(utils.locations.PATCHES_DIR)
                os.makedirs(utils.locations.PATCHES_DIR)



        bmask = self.utils.get_binary_mask(index)
        self.pimages = []

        images = self.utils.load_source_png_images(index)
        for j in images:
            self.pimages.append(Image.fromarray(j))

        histo = self.utils.load_histo_png_image(index)
        self.pimages.append(Image.fromarray(histo))
        print
        print 'Sampling subject %s, slice %d'%(self.subject, self.index)

        sampler = PatchSampler(bmask)
        selected = sampler.sample(PatchSampler.S_OVERLAP, params=None)
        #selected = sampler.sample(PatchSampler.S_MONTECARLO,params={'C':coverage})


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

        for i in range(L):
            label = self.utils.locations.LABELS[i]
            image = self.pimages[i]

            cimage = image.crop((x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
            #cimage.save(filename%(self.subject, self.index, label, x, y))  not saving intermediate images
            samples.append(np.array(cimage))

        self.generate_multi_channel_image(samples, x,y)

        image = self.pimages[-1]  #  the histology image

        cimage = image.crop((x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
        cimage.save(filename%(self.subject, self.index, 'HI', x,y))



        return L+1

    def generate_multi_channel_image(self, samples, x,y):

        N = len(samples)
        first = samples[0]
        (width, height, _) = first.shape
        multi = np.zeros((width,height,N),'uint8')
        for i in range(N):
            multi[...,i] =  samples[i][...,0]

        img = Image.fromarray(multi)

        filename = self.utils.locations.PATCH_TEMPLATE
        img.save(filename%(self.subject, self.index,'MU',x,y))






