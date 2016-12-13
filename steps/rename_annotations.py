import os

from deephisto import Locations

from config import dh_load_subjects, dh_read_config

#config = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
config = dh_read_config('/home/dcantor/projects/deephisto/code/config_field_fraction.ini')

locations = Locations(config)
subjects = dh_load_subjects(config)

for subject in subjects:
    locations.set_subject(subject)
    files = os.listdir(locations.MASK_DIR)
    print
    print locations.MASK_DIR
    for f in files:
        if f.startswith('A_'):
            A,B,C,I = f.split('_')
            I1,_ = I.split('.')
            change = 'A_H_%s.png'%I1

            source = locations.MASK_DIR + '/' + f
            target = locations.MASK_DIR + '/' + change
            print source,target
            os.rename(source, target)

