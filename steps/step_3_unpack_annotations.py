#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

from config import dh_load_subjects, dh_read_config
from deephisto import ImageUtils


def dh_unpack_annotations(config):
    """
    1. The generated Histology PNGs (H_xxx.png) are uploaded to  PLEXO (http://plexo.link) for every subject.
    2. In plexo the cortex is annotated so the patch creator knows where to generate patches from.
    3. The annotations must be saved in the annotations dir (see [png] section in config file).
    4. The zip file must be saved with the patient's id. For instance EPI_P027.zip
    To achieve this the series loaded in plexo must be named EPI_P027 (or the respective patient)

    Unpacking annotations is to extract the annotations contained in the ZIP file into the mask folder (see [png]
    mask_dir)

    Annotations are used for Quality Control (see step_4_visual_inspection)
    and subsequently to extract patches for training/testing the Convolutional Network.

    """
    utils = ImageUtils(config)
    subjects = dh_load_subjects(config)
    for s in subjects:
        utils.set_subject(s)
        utils.unpack_annotations()

def main():
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')
    dh_unpack_annotations(config)

if __name__=='__main__':
    main()

