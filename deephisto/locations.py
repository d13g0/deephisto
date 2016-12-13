#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

import os
import shutil
import string

from deephisto import Console


class FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class Locations:
    def __init__(self, config):

        self.config = config
        self.subject = None
        self.SOURCE_PNG = None
        self.HISTO_PNG = None
        self.LABELS_PNG = None
        self.HISTOLOGY_VOLUME = None
        self.SOURCES = None
        self.MASK_DIR = None
        self.ANNOTATIONS_ZIP = None

        # self.ROOT_DIR = root
        # self.ANNOTATIONS_DIR = None
        # self.ANNOTATIONS_ZIP = None
        # self.MASK_DIR = None
        # self.exv_dir = None
        # self.exv_FA = None
        # self.exv_MD = None
        # self.exv_MRI = None
        # self.SOURCES = None
        # self.LABELS = None
        # self.HIST_DIR = None
        # self.HIST_FMAP = None
        # self.IMAGES_DIR = None
        # self.SOURCE_PNG = None
        # self.HISTO_PNG = None
        #
        # if self.ROOT_DIR[-1] != '/':
        #     self.ROOT_DIR = self.ROOT_DIR + '/'
        #
        # self.MOVIE_DIR   = self.ROOT_DIR + 'movies'
        #
        # self.PATCHES_DIR = self.ROOT_DIR + 'patches'
        # self.PATCH_TEMPLATE = self.PATCHES_DIR + '/%s/P_%s_%d_%s_%d_%d.png'  # subject-slice-type-x-y.png
        #
        # self.CAFFE_WORKDIR = self.ROOT_DIR + 'caffe'
        # self.SPLIT_DIR = self.CAFFE_WORKDIR + '/split'
        #
        # self.TRAINING_TXT = self.SPLIT_DIR + '/%s/training.txt'
        # self.VALIDATION_TXT = self.SPLIT_DIR + '/%s/validation.txt'
        # self.TRAINING_AVERAGE = self.SPLIT_DIR + '/%s/training_average.txt'
        # self.TRAINING_AVG_IMAGE = self.SPLIT_DIR + '/%s/training_avg.png'
        #
        # if subject is not None:
        #     self.SUBJECT_DIR = '/subjects/%s/' % subject
        #     self.subject = subject
        #     self.update()
        # else:
        #     self.subject = None

    # def set_root_dir(self, root):
    #     self.ROOT_DIR = root
    #     self.update()

    @staticmethod
    def check_dir_of(path):
        if not os.path.exists(os.path.dirname(path)):
            print Console.WARNING + 'Creating directory %s' % os.path.dirname(path) + Console.ENDC
            os.makedirs(os.path.dirname(path))

    @staticmethod
    def create_empty_dir(path):
        if not os.path.exists(path):
            print Console.WARNING + 'Creating directory %s' % path + Console.ENDC
            os.makedirs(path)
        else:
            print Console.WARNING + 'Clearing %s' % path + Console.ENDC
            shutil.rmtree(path)
            os.makedirs(path)

    @staticmethod
    def partial_formatter(value, **kwargs):
        formatter = string.Formatter()
        mapping = FormatDict(kwargs)
        return formatter.vformat(value, (), mapping)

    def set_subject(self, subject):
        """
        Sets the current subject for path operations
        """
        f = Locations.partial_formatter
        self.subject    = subject
        self.SOURCE_PNG = f(self.config.SOURCE_PNG, subject=subject)
        self.HISTO_PNG  = f(self.config.HISTO_PNG,  subject=subject)
        self.LABELS_PNG = f(self.config.LABELS_PNG, subject=subject)
        self.HISTOLOGY_VOLUME = f(self.config.local_files.HI, subject=subject)
        self.IMAGES_DIR  = f(self.config.IMAGES_DIR, subject=subject)
        self.MASK_DIR    = f(self.config.MASK_DIR, subject=subject)
        self.SLICE_MASK_PNG = f(self.config.SLICE_MASK_PNG, subject=subject)
        self.ANNOTATIONS_ZIP = f(self.config.ANNOTATIONS_ZIP, subject=subject)

        self.SOURCES = [
            f(self.config.local_files.MR, subject=subject),
            f(self.config.local_files.FA, subject=subject),
            f(self.config.local_files.MD, subject=subject)
        ]

        self.TYPES = ['MRI', 'FA', 'MD']  #SAME ORDER AS IN SOURCES (MUST BE)



    def check_png_directories(self):
        """
        Checks that the PNG directories exists if not it creates them
        """
        check_dir_of = Locations.check_dir_of
        check_dir_of(self.HISTO_PNG)
        check_dir_of(self.LABELS_PNG)
        check_dir_of(self.SOURCE_PNG)

    def get_mask_png_location(self, index):
        """
        Returns the location of the slice mask for the requested index.
        It does not validate the index.
        A subject must be set prior to call this method
        """
        if self.subject == None:
            raise AssertionError('A subject must be set before calling this method')
        f = Locations.partial_formatter
        return f(self.SLICE_MASK_PNG, index=index)

    def get_source_png_location(self, type, index):
        """
        Returns the location of the requested source file.
         A subject must be set prior to call this method
        """
        if self.subject == None:
            raise AssertionError('A subject must be set before calling this method')
        f = Locations.partial_formatter
        return f(self.SOURCE_PNG,type=type,index=index)

    def get_label_png_location(self, index):
        """
        Returns the location of the requested label file.
        A subject must be set prior to call this method
        """
        if self.subject == None:
            raise AssertionError('A subject must be set before calling this method')
        f = Locations.partial_formatter
        return f(self.LABELS_PNG, index=index)

    def get_histo_png_location(self,index):
        """
        Returns the location of the requested histology file.
         A subject must be set prior to call this method
        """
        if self.subject == None:
            raise AssertionError('A subject must be set before calling this method')
        f = Locations.partial_formatter
        return f(self.HISTO_PNG,index=index)


    def update(self):
        pass

        # if self.ROOT_DIR[-1] != '/':
        #     self.ROOT_DIR = self.ROOT_DIR + '/'
        #
        # self.MOVIE_DIR = self.ROOT_DIR + 'movies'

        #

        #
        # # -----------------------------------------------------------------------------
        # #   Location of the histology feature maps
        # # -----------------------------------------------------------------------------
        # self.HIST_DIR = self.ROOT_DIR + self.SUBJECT_DIR + '/hist/'
        # self.HIST_FMAP = self.HIST_DIR + 'count_deformable_100um.nii.gz'
        #
        # # -----------------------------------------------------------------------------
        # #   Location of the resulting png images
        # # -----------------------------------------------------------------------------
        # self.IMAGES_DIR  = self.ROOT_DIR + self.SUBJECT_DIR + '/png/'
        # self.SOURCE_PNG  = self.IMAGES_DIR + 'S_%s_%d.png'
        # self.HISTO_PNG   = self.IMAGES_DIR + 'S_HI_%d.png'  # scaled histology used for annotations
        # self.HISTO_PNG_U = self.IMAGES_DIR + 'S_HU_%d.png'  #unscaled histology used for processing
        #
        #
        # # -----------------------------------------------------------------------------
        # #   Location of the patches
        # # -----------------------------------------------------------------------------
        # self.PATCHES_DIR = self.ROOT_DIR + 'patches'
        # self.PATCH_TEMPLATE = self.PATCHES_DIR + '/%s/P_%s_%d_%s_%d_%d.png'  # subject-slice-type-x-y.png
