# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:30:15 2016

@author: dcantor
"""

import deephisto as dh
from deephisto.caffe import *

SUBJECTS = []
ids = [27,31,32,33,34,36,37,40,44]
for i in ids:
    SUBJECTS.append('EPI_P0%s'%i)

locations = dh.ImageLocations('/home/dcantor/projects/deephisto')
utils = dh.ImageUtils(locations)

def demo_patch_search():
    visualizer = dh.PatchVisualizer(locations)
    visualizer.set_subject('EPI_P036')
    visualizer.set_slice(1)
    visualizer.demo_patch_search()


def step_1_download_nifti_files():
    """
    Downloads the NIFTI images (MRI, DTI, hISTO) from the remote location. Check the class ImageRetriever
    to configure these paths
    """
    dog = dh.ImageRetriever(locations)
    for s in SUBJECTS:
        dog.retrieve(s)


def step_2_create_all_pngs():
    """
    Creates PNGs for all the subjects in the subject list. The NIFTI images from where these PNGs are generated
    must have been downloaded first, and be accessible to the class ImageLocations
    """
    for s in SUBJECTS:
        print s
        utils.set_subject(s)
        utils.create_png_images()


def step_3_unpack_annotations():
    for s in SUBJECTS:
        utils.set_subject(s)
        utils.unpack_annotations()


def step_4_visual_inspection():
    """
    This step is repeated for every subject. The idea is to check that the registration between the
    histology and the exvivo images is accurate and that the annotations (histology mask) fits these images
    appropriately
    """
    visualizer = dh.PatchVisualizer(locations)
    visualizer.set_subject('EPI_P036') #replace with the subject to analyze (EPI_P27,EPI_P31, etc...32,33,34,36,37,40,44)
    visualizer.set_slice(1)
    visualizer.init()
    visualizer.create_patch(325,362)
    #visualizer.update_patch(250,100)


def step_5_create_patches():
    pcreator = dh.PatchCreator(utils)
    pcreator.create_patches('EPI_P036',1,cleardir=True,coverage=70)


def step_6_create_datasets():
    ds = dh.DatasetCreator(locations, training=0.7)
    ds.create()
    ds.get_average_training_set()


def step_7_create_dnn(): #dnn:deep neural network

    descriptor = NetworkDescriptor()
    descriptor.make(locations.ROOT_DIR+'caffe', locations.PATCHES_DIR)


def show_dnn_structure(model, snapshot):
    descriptor = NetworkDescriptor()
    descriptor.show_structure(model, snapshot)

def test():
    import matplotlib.pyplot as plt
    import Image
    import numpy as np
    PatchSampler = dh.PatchSampler
    utils.set_subject('EPI_P036')
    histo = utils.load_histo_png_image(3)
    x = 325
    y = 362
    im = Image.fromarray(histo).crop((x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
    im = np.array(im)[:, :, 0]

    numrows, numcols = im.shape
    print 'create patch histo (min, max) = (%d, %d)' % (im.min(), im.max())
    print 'rows, cols = (%d, %d)' % (numrows, numcols)

    def format_coord(x, y):
        row = int(x + 0.5)
        col = int(y + 0.5)

        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = im[row, col]
            return 'x=%d, y=%d, Neuronal Count =%d' % (row, col, z)
        else:
            return 'x=%d, y=%d' % (x, y)


    fig, ax = plt.subplots()
    imag = ax.imshow(im, interpolation='none',vmin=0, vmax=15)
    plt.colorbar(imag, ax=ax)
    ax.format_coord = format_coord
    plt.show()


#step_1_download_nifti_files()
#step_2_create_all_pngs()
#step_3_unpack_annotations()
step_4_visual_inspection()
#step_5_create_patches()
#step_6_create_datasets()
#step_7_create_dnn()

#demo_patch_search()



