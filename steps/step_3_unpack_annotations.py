from subjects import dh_load_subjects
from deephisto import ImageUtils, Locations


def dh_unpack_annotations(subjects, locations):
    """
    The Histology PNGs (S_HI_x.png) are uploaded to  PLEXO (http://plexo.link) for every subject
    to manually annotate the cortex.

    These annotations are saved as a ZIP file under the annotations folder for every patient
    (see Locations.ANNOTATIONS_DIR). This method unpack that zip file and places the individual
    annotations in the Locations.MASK_DIR folder.

    This is an important step, so DeepHisto can tell which slices from every patient are actually annotated and
    which ones are not.
    """
    utils = ImageUtils(locations)
    for s in subjects:
        utils.set_subject(s)
        utils.unpack_annotations()

def dh_get_histo_range(subjects, locations):
    utils = ImageUtils(locations)
    for s in subjects:
        print
        print s
        print '---------------------------------'
        utils.set_subject(s)
        utils.get_dynrange_histo()

if __name__=='__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    subjects = dh_load_subjects()
    dh_unpack_annotations(subjects, locations)
    dh_get_histo_range(subjects, locations)