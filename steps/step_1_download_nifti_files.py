#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor
#

from deephisto import ImageRetriever

from config import dh_load_subjects, dh_read_config



def main():
    """
    This is the first step you need to run on your DeepHisto project.
    Use one of the configuration files or create your own.

    >> config_neuronal_densitiy.ini
    >> config_field_fraction.ini

    """
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

    subjects = dh_load_subjects(config)
    dog = ImageRetriever(config)
    for s in subjects:
        dog.retrieve(s)

if __name__ == '__main__':
    main()