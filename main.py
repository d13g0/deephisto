# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:30:15 2016

@author: dcantor
"""
import os
os.environ['GLOG_minloglevel'] = '3'
from deephisto import Locations, ImageUtils, PatchSampler, PatchCreator, DatasetCreator
from deephisto.caffe import *
from visualizer import Visualizer
from sampling_demo import PatchSamplingDemo


# Add subjects here as they become available
#--------------------------------------------
ids = [27, 31, 32, 33, 34, 36, 37, 40, 44]
#--------------------------------------------

SUBJECTS = []

for i in ids:
    SUBJECTS.append('EPI_P0%s' % i)

locations = Locations('/home/dcantor/projects/deephisto')
utils = ImageUtils(locations)


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


def step_4_visual_inspection(subject, slice, x,y, wsize):
    visualizer = Visualizer(locations, wsize=wsize)
    visualizer.set_subject(subject)
    visualizer.set_slice(slice)
    visualizer.init()
    visualizer.create_patch(x, y)


def step_5_create_patches(target_dir):

    # sampler = PatchSampler(type=PatchSampler.TYPE_MONTECARLO,
    #                        params=dict(coverage=0.8))

    # ps = PatchSampler(type=PatchSampler.TYPE_CONVEX,
    #                  params=dict(overlap_factor=2))

    # sampler = PatchSampler(wsize=28, type=PatchSampler.TYPE_BACKGROUND,
    #                   params=dict(overlap_factor=2, xmax=3, ymax=3))

    sampler = PatchSampler(wsize=28,
                      type=PatchSampler.TYPE_OVERLAP,
                      params=dict(overlap_factor=8, edges=True, xcols=2, xrows=1))

    pcreator = PatchCreator(utils, sampler, target_dir)

    pcreator.create_patches('EPI_P036', 1, cleardir=True)

    for index in [3, 5, 7]:
        pcreator.create_patches('EPI_P036', index)

    for index in [2, 4, 6, 8]:
        pcreator.create_patches('EPI_P037', index)


def step_6_create_datasets(source_dir, training):
    ds = DatasetCreator(locations, training=training)
    ds.create_from(source_dir)


def step_7_create_dnn(net_name, wsize, data_dir):  # dnn:deep neural network
    nd = NetBuilder(net_name, wsize, data_dir=data_dir)
    nd.make()


def test():
    import matplotlib.pyplot as plt
    import Image
    import numpy as np
    PatchSampler = dh.PatchSampler
    utils.set_subject('EPI_P036')
    histo = utils.load_unscaled_histo_png_image(3)
    x = 325
    y = 362
    im = Image.fromarray(histo).crop(
            (x - PatchSampler.WSIDE, y - PatchSampler.WSIDE, x + PatchSampler.WSIDE, y + PatchSampler.WSIDE))
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
    imag = ax.imshow(im, interpolation='none', vmin=0, vmax=15)
    plt.colorbar(imag, ax=ax)
    ax.format_coord = format_coord
    plt.show()


def demo_patch_search():
    demo = PatchSamplingDemo(locations)
    # ps = PatchSampler(wsize=28,type=PatchSampler.TYPE_MONTECARLO,
    #                  params=dict(coverage=0.8, edges=True, xrows=2, xcols=2),
    #                  callback=demo.show_rectangle)
    ps = PatchSampler(wsize=28,
                      type=PatchSampler.TYPE_OVERLAP,
                      params=dict(overlap_factor=8, edges=True, xcols=2, xrows=1),
                      callback=demo.show_rectangle)
    # ps = PatchSampler(wsize=28, type=PatchSampler.TYPE_BACKGROUND,
    #                                    params=dict(overlap_factor=2, xmax=3, ymax=3),
    #                                    callback=demo.show_rectangle)
    demo.configure('EPI_P036', 7, ps)
    demo.run()



#--------------------------------------------
# STEPS (uncomment as required)
#--------------------------------------------
# step_1_download_nifti_files()

# step_2_create_all_pngs()

# step_3_unpack_annotations()

#demo_patch_search()


#step_4_visual_inspection(subject='EPI_P036',slice=1,x=391,y=335,wsize=28)

#step_5_create_patches('28x28_dense')

step_6_create_datasets('28x28_dense', training=0.7)

#step_7_create_dnn('dh28', 28, '28x28')

print
print '----'
print 'DONE'
