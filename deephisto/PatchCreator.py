import numpy as np
import Image
import os, shutil
from .PatchSampler import PatchSampler
from .Console import  Console


class PatchCreator:

    def __init__(self, utils):
        self.utils = utils
        self.subject = None
        self.index = None
        self.pimages = []


    def create_patches(self, subject, index, cleardir=False,coverage=30):
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
                print Console.WARNING + utils.locations.PATCHES_DIR + ' exists. Emptying the directory before creating new patches' + Console.ENDC
                shutil.rmtree(utils.locations.PATCHES_DIR)
                os.makedirs(utils.locations.PATCHES_DIR)



        bmask = self.utils.get_binary_mask(index)
        self.pimages = []

        images = self.utils.load_source_png_images(index)
        for j in images:
            self.pimages.append(Image.fromarray(j))

        histo = self.utils.load_histo_png_image(index)
        self.pimages.append(Image.fromarray(histo))

        sampler = PatchSampler(bmask)
        selected = sampler.sample(PatchSampler.S_MONTECARLO,params={'C':coverage})


        count = 0
        for (x,y) in selected:
            count = count  + self._extract_patch(x,y)

        print '-----------------------------------------------------------'
        print Console.OKBLUE + Console.BOLD + '%d'%count + Console.ENDC + ' png images were created'
        print '-----------------------------------------------------------'

    def _extract_patch(self, x, y):

        filename = self.utils.locations.PATCH_TEMPLATE

        L = len(self.pimages)-1

        samples = []

        for i in range(L):
            label = self.utils.locations.LABELS[i]
            image = self.pimages[i]
            cimage = image.crop((y - PatchSampler.WSIDE, x - PatchSampler.WSIDE, y + PatchSampler.WSIDE, x + PatchSampler.WSIDE))
            cimage.save(filename%(self.subject, self.index, label, y, x))
            samples.append(np.array(cimage))

        self.generate_multi_channel_image(samples, x,y)

        image = self.pimages[-1]  #  the histology image
        cimage = image.crop((y - PatchSampler.WSIDE, x - PatchSampler.WSIDE, y + PatchSampler.WSIDE, x + PatchSampler.WSIDE))
        cimage.save(filename%(self.subject, self.index, 'HI', y,x))



        return L+1

    def generate_multi_channel_image(self, samples, x,y):
        filename = self.utils.locations.PATCH_TEMPLATE
        N = len(samples)
        first = samples[0]
        (width, height, _) = first.shape
        multi = np.zeros((width,height,N),'uint8')
        for i in range(N):
            multi[...,i] =  samples[i][...,0]

        img = Image.fromarray(multi)
        img.save(filename%(self.subject, self.index,'MU',y,x))






