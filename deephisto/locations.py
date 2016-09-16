# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:28:53 2016

@author: dcantor
"""


class Locations:
    def __init__(self, root, subject=None):

        self.ROOT_DIR = root
        self.ANNOTATIONS_DIR = None
        self.ANNOTATIONS_ZIP = None
        self.MASK_DIR = None
        self.exv_dir = None
        self.exv_FA = None
        self.exv_MD = None
        self.exv_MRI = None
        self.SOURCES = None
        self.LABELS = None
        self.HIST_DIR = None
        self.HIST_FMAP = None
        self.IMAGES_DIR = None
        self.PNG_TEMPLATE = None
        self.PNG_TEMPLATE_H = None

        if self.ROOT_DIR[-1] != '/':
            self.ROOT_DIR = self.ROOT_DIR + '/'

        self.PATCHES_DIR = self.ROOT_DIR + 'patches'
        self.PATCH_TEMPLATE = self.PATCHES_DIR + '/P_%s_%d_%s_%d_%d.png'  # subject-slice-type-x-y.png

        self.CAFFE_WORKDIR = self.ROOT_DIR + 'caffe'
        self.SPLIT_DIR = self.CAFFE_WORKDIR + '/split'
        self.TRAINING_TXT = self.SPLIT_DIR + '/training.txt'
        self.VALIDATION_TXT = self.SPLIT_DIR + '/validation.txt'
        self.TRAINING_AVERAGE = self.SPLIT_DIR + '/training_average.txt'
        self.TRAINING_AVG_IMAGE = self.SPLIT_DIR + '/training_avg.png'

        if subject is not None:
            self.SUBJECT_DIR = '/subjects/%s/' % subject
            self.subject = subject
            self.update()
        else:
            self.subject = None

    def set_root_dir(self, root):
        self.ROOT_DIR = root
        self.update()

    def set_subject(self, subject):
        self.SUBJECT_DIR = '/subjects/%s/' % subject
        self.subject = subject
        self.update()

    def get_mask_location(self, mask_index):
        return self.MASK_DIR + 'A_[S_%d_t_HI].png' % mask_index

    def update(self):

        if self.ROOT_DIR[-1] != '/':
            self.ROOT_DIR = self.ROOT_DIR + '/'
        # -----------------------------------------------------------------------------
        #   Location of the annotated images
        # -----------------------------------------------------------------------------
        self.ANNOTATIONS_DIR = self.ROOT_DIR + self.SUBJECT_DIR + '/annotations'
        self.ANNOTATIONS_ZIP = self.ANNOTATIONS_DIR + '/%s.zip' % self.subject
        self.MASK_DIR = self.ROOT_DIR + self.SUBJECT_DIR + '/mask/'

        # -----------------------------------------------------------------------------
        #   Location of the ex-vivo images
        # -----------------------------------------------------------------------------
        self.exv_dir = self.ROOT_DIR + self.SUBJECT_DIR + '/exvivo/'
        self.exv_MRI = self.exv_dir + 'reg_ex_mri_100um.nii.gz'
        self.exv_FA = self.exv_dir + 'dti_FA.100um.nii.gz'
        self.exv_MD = self.exv_dir + 'dti_MD.100um.nii.gz'
        self.SOURCES = [self.exv_MRI, self.exv_FA, self.exv_MD]
        self.LABELS = ['MRI', 'FA', 'MD']

        # -----------------------------------------------------------------------------
        #   Location of the histology feature maps
        # -----------------------------------------------------------------------------
        self.HIST_DIR = self.ROOT_DIR + self.SUBJECT_DIR + '/hist/'
        self.HIST_FMAP = self.HIST_DIR + 'count_deformable_100um.nii.gz'

        # -----------------------------------------------------------------------------
        #   Location of the resulting png images
        # -----------------------------------------------------------------------------
        self.IMAGES_DIR = self.ROOT_DIR + self.SUBJECT_DIR + '/png/'
        self.PNG_TEMPLATE = self.IMAGES_DIR + 'S_%d_t_%s.png'
        self.PNG_TEMPLATE_H = self.IMAGES_DIR + 'S_%d_t_HI.png'

        # -----------------------------------------------------------------------------
        #   Location of the patches
        # -----------------------------------------------------------------------------
        self.PATCHES_DIR = self.ROOT_DIR + 'patches'
        self.PATCH_TEMPLATE = self.PATCHES_DIR + '/P_%s_%d_%s_%d_%d.png'  # subject-slice-type-x-y.png
