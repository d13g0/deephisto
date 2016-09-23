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

if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    subjects  = dh_load_subjects()
    dh_create_pngs(subjects, locations)