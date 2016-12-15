#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

from deephisto import PatchSampler, PatchCreator, Locations, ImageUtils

from config import dh_load_subjects, dh_read_config


def dh_get_slices(config):
    slices = {}
    utils = ImageUtils(config)
    subjects = dh_load_subjects(config)

    print
    print 'List of subjects [slices] that will be used to create patches '
    print '--------------------------------------------------------------'
    for s in subjects:
        utils.set_subject(s)
        slices[s] = utils.get_annotation_indices()
        print s, slices[s]

    return slices


def dh_create_patches(slices, config):

    utils = ImageUtils(config)

    cortex_sampler = PatchSampler(wsize=config.PATCH_SIZE, type=PatchSampler.TYPE_MONTECARLO, params=dict(coverage=0.7))

    backgr_sampler = PatchSampler(wsize=config.PATCH_SIZE, type=PatchSampler.TYPE_BACKGROUND,
                                      params=dict(overlap_factor=2, xmax=3, ymax=3))

    cortex_creator = PatchCreator(cortex_sampler, utils, config)
    backgr_creator = PatchCreator(backgr_sampler, utils, config)

    cortex_creator.clear_dir(True)

    for subject in sorted(slices):
        indices = slices[subject]
        for index in indices:
            cortex_creator.create_patches(subject, index)
            backgr_creator.create_patches(subject, index)


def main():
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

    slices = dh_get_slices(config)
    dh_create_patches(slices, config)


if __name__ == '__main__':
    main()
