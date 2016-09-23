from subjects import dh_load_subjects
from deephisto import ImageRetriever, Locations


def dh_download_nifti_files(subjects, loc):
    """
    Downloads the NIFTI images (MRI, DTI, HISTO) from the remote location.
    """

    root = loc.ROOT_DIR

    # %s will be replaced by EPI_PXXX according to the SUBJECTS list (see subjects.py)
    remote_exvivo_dir = '/home/dcantor/remote/epilepsy/%s/Processed/Ex-Hist_Reg/9.4T/Neo/aligned_Ex_100um/'
    remote_histo_dir = '/home/dcantor/remote/histology/Histology/%s/100um_5umPad_FeatureMaps/aligned/Neo_NEUN/'
    sources = {     # a map indicating the remote locations ofr these files
        'FA': remote_exvivo_dir + 'dti_smoothed_0.2/dti_FA.100um.nii.gz',
        'MD': remote_exvivo_dir + 'dti_smoothed_0.2/dti_MD.100um.nii.gz',
        'MR': remote_exvivo_dir + 'reg_ex_mri_100um.nii.gz',
        'HI': remote_histo_dir + 'count_deformable_100um.nii.gz'
    }

    local_exvivo_dir = root + '/subjects/%s/exvivo/'
    local_histo_dir = root + '/subjects/%s/hist/'
    targets = {     # a map indicating where the remote files will be copied
        'FA': local_exvivo_dir + 'dti_FA.100um.nii.gz',
        'MD': local_exvivo_dir + 'dti_MD.100um.nii.gz',
        'MR': local_exvivo_dir + 'reg_ex_mri_100um.nii.gz',
        'HI': local_histo_dir + 'count_deformable_100um.nii.gz'

    }


    dog = ImageRetriever(loc, sources, targets)
    for s in subjects:
        dog.retrieve(s)


if __name__=='__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    subjects = dh_load_subjects()
    dh_download_nifti_files(subjects, locations)