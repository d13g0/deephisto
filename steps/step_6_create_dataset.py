from deephisto import Locations, DatasetCreator


def dh_create_dataset(source_dir, training):
    ds = DatasetCreator(locations, training=training)
    ds.create_from(source_dir)

if __name__=='__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    dh_create_dataset('28x28e', 0.7)

