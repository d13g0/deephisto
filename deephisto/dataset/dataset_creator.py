import csv, glob, os.path

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from sklearn.cross_validation import  ShuffleSplit

from deephisto.utils.console import Console


class DatasetCreator:

    def __init__(self, locations, training=0.7):
        self.locations = locations
        if training <= 0 or training > 1:
            print Console.FAIL + 'Training percercentage must be in the (0,1] range' + Console.ENDC
            raise AssertionError

        self.training = training  #number from 0 to 1 percentage of the dataset dedicated to training
        self.validation = 1 - training  #the complement to 1 of the training split

    def create_from(self, DS_DIR):
        data_dir = self.locations.PATCHES_DIR + '/' + DS_DIR
        route = data_dir + '/*.png'
        print
        print 'Creating datasets'
        print '-----------------'
        print 'Route: %s'%route
        print 'Training percentage      : %d%%'% (self.training * 100)
        print 'Validation percentage    : %d%%'% (self.validation *100)

        files = glob.glob(route)


        sources = [os.path.basename(f) for f in files if '_MU_' in f]
        targets = []

        if len(sources) == 0:
            raise Exception('The source directory is empty')

        dict ={}

        for s in sources:
            _,_,P,slice,_,x,y = s.split('_') #P_EPI_PXXX_1_TYPE_X_Y.png
            patient = 'EPI_%s'%P
            y,_ = y.split('.')
            target_path = self.locations.PATCH_TEMPLATE%(DS_DIR, patient,int(slice),'HI',int(x),int(y)) #/P_%s_%d_%s_%d_%d.png'
            target = os.path.basename(target_path)
            if os.path.isfile(target_path):
                exists = True
                targets.append(target)
            else:
                exists = False
                print s + ' -> ' + target + '  '+ Console.FAIL + '[fail]' + Console.ENDC

            if not exists:
                print 'There are missing files'
                return


        # this is why I love python (well, one of the reasons...)
        map = zip(sources,targets)
        idx = 0
        N = len(map)
        train_indices = None
        validation_indices = None
        print 'Number of elements       : %d'%N

        split = ShuffleSplit(N,1, train_size=self.training, test_size=None)
        for a, b in split:
            train_indices = a
            validation_indices = b

        print
        print 'Database size            : %s'% (Console.BOLD + Console.OKBLUE + str(N) + Console.ENDC)
        print 'Training dataset size    : %s'%Console.OKBLUE + str(len(train_indices)) + Console.ENDC
        print 'Validation dataset size  : %s'%Console.OKBLUE + str(len(validation_indices)) + Console.ENDC
        print

        assert len(train_indices) + len(validation_indices) == N, "The dataset was not splitted adequately"

        training_txt = self.locations.TRAINING_TXT%DS_DIR

        self.locations.check_dir_of(training_txt)

        with open(training_txt,'wb') as training_file:
            writer = csv.writer(training_file, delimiter=';')
            for idx in train_indices:
                writer.writerow([map[idx][0], map[idx][1]])

        print training_txt + ' has been written'


        validation_txt = self.locations.VALIDATION_TXT%DS_DIR
        with open(validation_txt,'wb') as validation_file:
            writer = csv.writer(validation_file, delimiter=';')
            for idx in validation_indices:
                writer.writerow([map[idx][0], map[idx][1]])

        print validation_txt + ' has been written'
        print

        self._compute_average(DS_DIR)

    def _compute_average(self, DS_DIR):

        training_txt = self.locations.TRAINING_TXT%DS_DIR
        with open(training_txt,'r') as training_file:

            pairs = training_file.read().splitlines()
            N = len(pairs)

            print
            print 'Calculating average RGB image and value in training set'
            print '-------------------------------------------------------'
            print 'Number of elements in training set %d'%N

            mean_image = None
            mean_all = np.array([0, 0, 0])

            DATA_DIR = self.locations.PATCHES_DIR + '/' + DS_DIR

            for p in pairs:
                source, target = p.split(';')
                im = np.array(Image.open(DATA_DIR + '/' + source))

                avg_r = im[:,:,0].mean()
                avg_g = im[:,:,1].mean()
                avg_b = im[:,:,2].mean()
                mean_im = np.array([avg_r, avg_b, avg_b], dtype=np.int)
                mean_all += mean_im


                if mean_image is None:
                    mean_image = np.copy(im).astype(np.int64)
                else:
                    mean_image = np.add(mean_image,im)

            mean_image = np.rint(mean_image / N).astype(np.uint8)
            print 'Mean image:'
            print '-----------'
            #print mean_image.dtype, mean_image.min(), mean_image.max()
            print 'R: %.2f'%mean_image[:, :, 0].mean()
            print 'G: %.2f'%mean_image[:, :, 1].mean()
            print 'B: %.2f'%mean_image[:, :, 2].mean()
            plt.imshow(mean_image,interpolation='None')
            plt.show()


            training_avg_img_file = self.locations.TRAINING_AVG_IMAGE%DS_DIR

            Image.fromarray(mean_image,'RGB').save(training_avg_img_file)
            print 'Mean image saved to %s'%training_avg_img_file


            print
            print
            mean_all = np.array(mean_all)
            mean_all = np.rint(mean_all/float(N)).astype(int)
            print 'Average Value: %s'%mean_all
            print '-----------------------------'

            training_avg_file = self.locations.TRAINING_AVERAGE%DS_DIR

            with open(training_avg_file,'w') as avg_file:
                avg_file.write('%s'%mean_all)
            print 'Average value saved to %s'%training_avg_file





