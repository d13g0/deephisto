#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

from deephisto import Locations
from deephisto.image import Visualizer

from config import dh_load_subjects, dh_read_config


def dh_visual_inspection(config, subject, x, y):
    visualizer = Visualizer(config)
    visualizer.set_subject(subject)
    visualizer.init()
    visualizer.create_patch(x, y)


def dh_inspect_slice(config, subject, index, x, y):
    """
    In case you want to start the visualizer at a given slice

    for example dh_inspect_slice(config,'EPI_P027',3,334,220)
    """
    visualizer = Visualizer(config)
    visualizer.set_subject(subject)
    visualizer.init()
    visualizer.set_slice(index)
    visualizer.create_patch(x, y)


def main():
    # config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

    subjects = dh_load_subjects(config)
    for s in subjects:
        dh_visual_inspection(config, s, 0, 0)

if __name__ == '__main__':
    main()
