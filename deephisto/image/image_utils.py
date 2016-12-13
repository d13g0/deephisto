#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor
#

import errno
import os
import shutil
import zipfile

import Image
import nibabel as nib
import numpy as np
from PIL import ImageFilter
from deephisto import Console, Locations
from scipy import misc
from scipy.misc import bytescale


class ImageUtils:


    @staticmethod
    def load_nifti_image(filename):
        """
        Loads a nifti image and returns the data as a multidimensional numpy array
        """
        img = nib.load(filename)
        data = img.get_data()
        return data

    @staticmethod
    def data_to_labels_rgb(data, NUM_LABELS, RANGE_MIN, RANGE_MAX):
        """
        Rescales the histology for deep learning processing.
        Since labels start at zero, the high parmeter is NUM_LABELS-1

        the cmin and cmax parameters depend on the properties of the histology
        and can be changed to obtain a different linear mapping of the final labels

        Example:

        test = np.linspace(0,1,num=100)

        array([ 0.        ,  0.01010101,  0.02020202,  0.03030303,  0.04040404,
                0.05050505,  0.06060606,  0.07070707,  0.08080808,  0.09090909,
                ...

                0.95959596,  0.96969697,  0.97979798,  0.98989899,  1.        ])

        bytescale(test,low=0, high=15, cmin=0,cmax=0.5)
        array([ 0,  0,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,
        5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9, 10, 10,
       10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], dtype=uint8)

        bytescale(test,low=0, high=15-1, cmin=0,cmax=0.5)
        array([ 0,  0,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  5,
        5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,
       10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14,
       14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
       14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
       14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14], dtype=uint8)

        """
        im = bytescale(data,low=0, high=NUM_LABELS-1, cmin=RANGE_MIN, cmax=RANGE_MAX)
        w, h = data.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = im
        ret[:, :, 1] = im
        ret[:, :, 2] = im
        return ret

    @staticmethod
    def data_to_bytescale_rgb(data):
        """
        Converts a single-channel grayscale image to a 3-channel image that can be 
        then saved as a  PNG        
        """
        im = bytescale(data)
        w, h = im.shape
        ret = np.empty((w,h,3), dtype=np.uint8)
        ret[:,:,0] = im
        ret[:,:,1] = im
        ret[:,:,2] = im
        return ret


    def __init__(self, config):
        self.config = config
        self.locations = Locations(config)
        self.subject = self.locations.subject


    def load_histology_volume(self):
        """
        Loads the histology volume. Set subject must have been called previously
        """
        if self.subject is None:
            raise Exception('set_subject must be called first')

        return ImageUtils.load_nifti_image(self.locations.HISTOLOGY_VOLUME)


    def load_source_volumes(self):
        """
        Returns an array with the nifti volumes corresponding to the source images
         (MR, FA, MD)
        """
        volumes = []
        try:
            for s in self.locations.SOURCES:
                volumes.append(ImageUtils.load_nifti_image(s))
        except IOError as e:
            print Console.FAIL + 'There are errors loading nifi files for subject %s' % self.subject + Console.ENDC
            raise IOError(e)
        return volumes


    def set_subject(self, subject):
        """
        Sets the current subject for image operations
        """
        self.subject = subject
        self.locations.set_subject(subject)


    def create_png_images(self, subject):
        """
        Creates the initial set of PNG images. Each PNG image corresponds to a whole slice and it can be
        MRI, Histology or Label.

        The label PNGs are a rescaled version of the Histology. Used for training/testing the deep learning
        network.

        Scaling the Histology is necessary as the network learns integer labels. So regardless
        of the original scale of the histology, the corresponding label image contains the version of
        the histology rescaled to an integer range.

        This range is predetermined in the configuration file under the section [classification]
        as:
        NUM_LABELS (please check config_neuronal_density.ini for an example).

        The preprocessing of the input images with a gaussian filter is an optional step.
        Please refer to the properties GAUSSIAN_BLUR and GAUSSIAN_RADIOUS in the section [classification]
        of the configuration file.

        """

        print 'Generating PNG files for ' +Console.BOLD+Console.OKBLUE+ subject + Console.ENDC
        print '\tGaussian blur  :\t%s'%self.config.GAUSSIAN_BLUR
        if self.config.GAUSSIAN_BLUR:
            print '\tGaussian radius:\t%.1f'%self.config.GAUSSIAN_RADIUS

        self.set_subject(subject)
        self.locations.check_png_directories()

        histology_volume = self.load_histology_volume()
        input_volumes = self.load_source_volumes()

        assert input_volumes[0].shape[2] == \
               input_volumes[1].shape[2] == \
               input_volumes[2].shape[2], 'Input NIFTI volumes must have the same number of slices'

        NUM_SLICES = input_volumes[0].shape[2]  # all the volumes must have the same number of slices

        self.locations.create_empty_dir(self.locations.IMAGES_DIR)

        print '\tMRI'
        for volume_idx, volume in enumerate(input_volumes):
            for index in range(NUM_SLICES):
                type = self.locations.TYPES[volume_idx]
                imslice = ImageUtils.data_to_bytescale_rgb(volume[:, :, index])
                im = Image.fromarray(imslice)

                im.save(self.locations.get_source_png_location(type, index))

        print '\tHistology'
        for index in range(NUM_SLICES):
            im_histo = ImageUtils.data_to_bytescale_rgb(histology_volume[:, :, index]);
            im_histo = Image.fromarray(im_histo)
            
            if self.config.GAUSSIAN_BLUR:
                im_histo = im_histo.filter(ImageFilter.GaussianBlur(radius=self.config.GAUSSIAN_RADIUS))  # Filter requested by Ali Khan
            
            im_histo.save(self.locations.get_histo_png_location(index))
                          

        print '\tLabels'
        for index in range(NUM_SLICES):

            im_labels = ImageUtils.data_to_labels_rgb(
                    histology_volume[:, :, index],
                    self.config.NUM_LABELS,
                    self.config.HISTOLOGY_RANGE_MIN,
                    self.config.HISTOLOGY_RANGE_MAX);
            im_labels = Image.fromarray(im_labels)
            
            if self.config.GAUSSIAN_BLUR:
                im_labels = im_labels.filter(ImageFilter.GaussianBlur(radius=self.config.GAUSSIAN_RADIUS))  # Filter requested by Ali Khan
            
            im_labels.save(self.locations.get_label_png_location(index))
                           

        print


    def get_annotation_indices(self):
       files = [fn for fn in os.listdir(self.locations.MASK_DIR) if fn.endswith('.png')]
       indices = []
       for fn in files:
           index  = int(fn.split('_')[2].split('.')[0])  #A_H_5.png = A, H, 5.png
           indices.append(index)
           indices.sort()
       return indices


    def get_dynrange_histology(self,annotated_only=False):
        """
        Gets the dynamic range for the histology volumes. The range can be reported over
         the whole histology volume or just over those slices that have been annotated

        This range can help deciding how many labels to set in the configuration:

        [classification]
        NUM_LABELS = ?

        also it can help determining the range for the histology values:

        [histology]
        HISTOLOGY_RANGE_MIN = ?
        HISTOLOGY_RANGE_MAX = ?

        :return: [min, max, mean] array of the histology for the current subject

        """
        if self.subject is None:
            print Console.WARNING + 'You need to specify a subject first' + Console.ENDC
            return

        volume = self.load_histology_volume()
        if not annotated_only:
            return volume.min(), volume.max()
        else:
            indices = self.get_annotation_indices()
            #slice number is the last index of the volume
            return  volume[:,:,indices].min(), volume[:,:,indices].max()


    def range_to_label(self, min,max):
        """
        Returns the respective label range according to the current configuration
        :return:
        """
        range = np.array([min,max])
        out = bytescale(range,
                       low=0,
                       high=self.config.NUM_LABELS - 1,
                       cmin=self.config.HISTOLOGY_RANGE_MIN,
                       cmax=self.config.HISTOLOGY_RANGE_MAX)
        return out.min(), out.max()


    def unpack_annotations(self):
        print self.locations.ANNOTATIONS_ZIP, self.locations.MASK_DIR
        # @TODO: This code works. Uncomment
        # url = self.locations.ANNOTATIONS_ZIP
        # try:
        #     shutil.rmtree(self.locations.MASK_DIR)
        # except OSError as e:
        #     if e.errno == errno.ENOENT:
        #         pass
        #
        # os.makedirs(self.locations.MASK_DIR)
        #
        # try:
        #     zipf = zipfile.ZipFile(url)
        #     zipf.extractall(self.locations.MASK_DIR)
        #     print Console.OKBLUE + 'All annotations have been extracted for %s'%self.subject + Console.ENDC
        # except IOError as e:
        #     if e.errno != errno.ENOENT: raise e
        #     else:
        #         print 'No annotations yet for %s'%self.subject

  
    def load_mask_for_slice(self, index):
        if self.subject is None:
            raise AssertionError('You need to specify a subject first')

        return misc.imread(self.locations.get_mask_png_location(index))
        

    def load_binary_mask_for_slice(self, index):
        """
        Reads a PNG mask (color) and returns a binary mask where elements in the mask
        are set to 255 and elements in the background are set to 0
        
        index:  index of the mask to retrieve
        """
        if self.subject is None:
            raise AssertionError('You need to specify a subject first')
        
        mask = self.load_mask_for_slice(index)
        (rows,cols) = np.where(mask>0)[0:2] #pixels in mask disregarding the color
        new_mask = np.zeros(shape=mask.shape[0:2], dtype=np.uint8)
        new_mask[(rows,cols)] = 255
        return new_mask

 
    def load_sources_for_slice(self, index):
        """
        Loads the respective source png images for the slice being analyzed
        Returns an array with the source PNGs for the requested index.
        The order in this array is the same order of location.TYPES
        A subject must be set before calling this method
        """
        if self.subject is None:
            raise AssertionError('You need to specify a subject first')

        
        data = []    
        for type in self.locations.TYPES:
            slice_data = misc.imread(self.locations.get_source_png_location(type,index))
            data.append(slice_data)
            
        return data #images in the same order as labels

    def load_labels_for_slice(self, index):
        """
        Loads the labels png for the requested slice
        A subject must be set before calling this method
        """
        if self.subject is None:
            raise AssertionError('You need to specify a subject first')

        return misc.imread(self.locations.get_label_png_location(index)) #png histology




    def load_multichannel_input(self, num_slice):
        """
        Creates a multi-channel array with all the inputs:
         R->MRI
         G->FA
         B->MD
         for now..
        """
        data = np.array(self.load_sources_for_slice(num_slice))
        items, w, h, channels = data.shape
        multi = np.empty((w,h,channels))
        assert items == channels, 'The number of images must be equal to the number of channels'
        for i in range(0,items):
            multi[:,:,i] = data[i,:,:,0]
        return multi





    def load_bytescaled_histo_png_image(self, num_slice):
        """
        Loads the respective Histology png for the slice being analyzed
        """
        if self.subject is None:
            print Console.WARNING + 'You need to specify a subject first' + Console.ENDC
            return
        data = []
        hfile = self.locations.HISTO_PNG % (num_slice)
        data = misc.imread(hfile)

        return data  # png histology

 




