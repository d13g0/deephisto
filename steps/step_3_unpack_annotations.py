#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

from config import dh_load_subjects, dh_read_config
from deephisto import ImageUtils, Locations


def dh_unpack_annotations(config):
    """
    The Histology PNGs (S_H_x.png) are uploaded to  PLEXO (http://plexo.link) for every subject
    to manually annotate the cortex.

    These annotations are saved as a ZIP file under the annotations folder for every patient
    (see Locations.ANNOTATIONS_DIR). This method unpack that zip file and places the individual
    annotations in the Locations.MASK_DIR folder.

    This is an important step, so DeepHisto can tell which slices from every patient are actually annotated and
    which ones are not.
    """
    utils = ImageUtils(config)
    subjects = dh_load_subjects(config)
    for s in subjects:
        utils.set_subject(s)
        utils.unpack_annotations()

def main():
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')
    dh_unpack_annotations(config)

if __name__=='__main__':
    main()

