from subjects import dh_load_subjects
from deephisto import Locations, ImageUtils


def dh_create_pngs(subjects, locations):
    """
    Creates PNGs for all the subjects in the subject list. The NIFTI images from where these PNGs are generated
    must have been downloaded first, and be accessible to the class ImageLocations
    """
    utils = ImageUtils(locations)
    for s in subjects:
        print s
        utils.set_subject(s)
        utils.create_png_images()


def dh_get_histo_range(subjects, locations):
    utils = ImageUtils(locations)
    for s in subjects:
        print
        print s
        print '---------------------------------'
        utils.set_subject(s)
        utils.get_dynrange_histo()

if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    subjects  = dh_load_subjects()
    dh_create_pngs(subjects, locations)
    dh_get_histo_range(subjects, locations)