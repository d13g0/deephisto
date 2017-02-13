#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor
from config import dh_read_config
from deephisto import Locations
from deephisto.net import NetBuilder_fcn8


def main():
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    #config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density_rt.ini')
    net = NetBuilder_fcn8(config)
    net.make()

if __name__ == '__main__':
    main()

