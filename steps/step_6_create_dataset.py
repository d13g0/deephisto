#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

from deephisto import Locations, DatasetCreator

from config import dh_read_config

def dh_create_dataset(config):
    ds = DatasetCreator(config)
    ds.run()

def main():
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

    dh_create_dataset(config)

if __name__=='__main__':
    main()

