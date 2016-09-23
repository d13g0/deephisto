
# Add subjects here as they become available
#--------------------------------------------
ids = [27, 31, 32, 33, 34, 36, 37, 40, 41, 44, 45, 46]
#--------------------------------------------

def dh_load_subjects():
    subjects = []
    for i in ids:
        subjects.append('EPI_P%03d' % i)
    return subjects


