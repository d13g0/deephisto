#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor
#

from deephisto import ImageUtils, Console

from config import dh_load_subjects, dh_read_config


def dh_get_histo_range(config):
    """
    The reported histology value range should provide a good estimate for the
    HISTOLOGY_RANGE_MIN and HISTOLOGY_RANGE_MAX that yoou can set under [histology] in the configuration file
    """
    utils = ImageUtils(config)
    subjects = dh_load_subjects(config)
    print 'Number of labels :'+Console.BOLD+ '%s'%config.NUM_LABELS+Console.ENDC
    print
    print 'subject\t\thistology domain\t\tlabel range'
    print '-------\t\t----------------\t\t-----------'
    for s in subjects:
        utils.set_subject(s)
        min, max = utils.get_dynrange_histology(annotated_only=False)
        min_l, max_l = utils.range_to_label(min, max)
        print '%s\t\t[%.2f,%.2f]\t\t     [%d,%d]'%(s,min,max,min_l,max_l)
    print


def main():
    """
    utils.create_png_images:

    Creates PNGs for all the subjects in the subject list. The NIFTI images from where these PNGs are generated
    must have been downloaded first (step 1).

    Make sure that you set NUM_LABELS, HISTOLOGY_RANGE_MIN and HISTOLOGY_RANGE_MAX in the configuration file before
    creating running this step.

    The HISTOLOGY_RANGE_MIN and HISTOLOGY_RANGE_MAX parameters determine how histology values are mapped to labels.
    These two parameters correspond to cmin, and cmax in scipy.misc.bytescale
    See: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.bytescale.html

    """

    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

    subjects = dh_load_subjects(config)
    utils = ImageUtils(config)

    dh_get_histo_range(config)


    # comment this for loop if you only want to run dh_get_histo_range. This will report the domain values (histology)
    for s in subjects:
        print s
        utils.create_png_images(s)

if __name__ == '__main__':
    main()
