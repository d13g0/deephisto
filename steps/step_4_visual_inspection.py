from subjects import dh_load_subjects
from interact import Visualizer
from deephisto import Locations



def dh_visual_inspection(subject, x,y, wsize):
    visualizer = Visualizer(locations, wsize=wsize)
    visualizer.set_subject(subject)
    visualizer.init()
    visualizer.create_patch(x, y)


if __name__=='__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    subjects = dh_load_subjects()
    for s in subjects:
        dh_visual_inspection(s, 0,0,28)

