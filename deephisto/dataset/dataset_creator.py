import csv, glob, os.path, pdb

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
#from sklearn.cross_validation import  ShuffleSplit
from random import shuffle

from deephisto.utils.console import Console
from deephisto import Locations


class DatasetCreator:

    def __init__(self, config):

        self.config = config
        self.locations = Locations(config)
        training = config.TRAINING_PERCENTAGE

        if training <= 0 or training > 1:
            print Console.FAIL + 'Training percercentage must be in the (0,1] range' + Console.ENDC
            raise AssertionError

        self.training = training  #number from 0 to 1 percentage of the dataset dedicated to training
        self.validation = 1 - training  #the complement to 1 of the training split

    def run(self):

        patch_dir = self.config.PATCH_DIR

        print
        print 'Dataset creation'
        print '-----------------'
        print 'Patch directory : %s'%patch_dir
        print 'Training percentage      : %d%%'% (self.training * 100)
        print 'Validation percentage    : %d%%'% (self.validation *100)
        print
        print Console.UNDERLINE + 'Please be patient while the patches directory is analyzed' + Console.ENDC
        files = glob.glob(patch_dir + '/*.png')
        print 'done.'

        sources = [os.path.basename(f) for f in files if os.path.basename(f).startswith('I_')] #review this
        L = len(sources)
        print
        print 'Number of input patches : '+ Console.OKBLUE + '%d'%L + Console.ENDC
        targets = []

        if len(sources) == 0:
            raise Exception('The source directory is empty')

        dict ={}

        for s in sources:
            # @TODO (hardcode) if patch_template in [patches] change, this code is affected
            _,_,P,slice,x,y = s.split('_') #{type}_{subject}_{index}_{x}_{y}.png
            subject = 'EPI_%s'%P
            y,_ = y.split('.')

            #check that the label file exists
            target_path = self.config.PATCH_TEMPLATE.format(subject=subject, index=int(slice), type='L', x=int(x), y=int(y))
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


        map = zip(sources,targets)

        #separate training and validation sets

        slices = {}
        """
        This loop creates an array of slices. The name of the slice is the 
        subject name + slice number.
        """
        for input, label in map:
            slice_key = '_'.join(input.split('_')[1:4])  #@TODO (hardcode) depends on patch_template under [patches]
            if not slice_key in slices:
                slices[slice_key] = [(input,label)]
            else:
                slices[slice_key].append((input,label))

        keys = slices.keys()
        M = len(keys)

        keys.sort()
        shuffle(keys)
        shuffle(keys)
        shuffle(keys)

        threshold = int(L  * self.training)

        sum = 0
        training_patches = []
        validation_patches = []
        idx = 0

        print
        print 'Training patches'
        print '----------------'
        while sum < threshold:
            slice_key = keys[idx]
            training_patches += slices[slice_key]
            sum += len(slices[slice_key])
            print slice_key, len(slices[slice_key])
            idx +=1
        print
        print 'Total : %d'%sum


        print
        print 'Validation patches'
        print '------------------'
        sum = 0
        for j in range(idx, M):
            slice_key = keys[j]
            validation_patches += slices[slice_key]
            sum += len(slices[slice_key])
            print slice_key, len(slices[slice_key])

        print 'Total : %d' % sum



        assert len(training_patches) + len(validation_patches) == L, 'The dataset was not splitted adequately'

        print
        print 'Database size            : %s'% (Console.BOLD + Console.OKBLUE + str(L) + Console.ENDC)
        print 'Estimated training size  : ' + Console.BOLD + '%d' % threshold + Console.ENDC
        print 'Training dataset size    : %s'%Console.OKBLUE + str(len(training_patches)) + Console.ENDC
        print 'Validation dataset size  : %s'%Console.OKBLUE + str(len(validation_patches)) + Console.ENDC
        print

        DATASET_DIR = self.config.DATASET_DIR
        self._write_files(training_patches, validation_patches, DATASET_DIR)
        self._compute_average(DATASET_DIR)



    def _write_files(self, training_patches, validation_patches, DATASET_DIR):

        self.locations.check_dir_of(self.config.TRAINING_FILE)

        with open(self.config.TRAINING_FILE,'wb') as training_file:
            writer = csv.writer(training_file, delimiter=';')
            for idx in training_patches:
                writer.writerow([idx[0], idx[1]])

        print self.config.TRAINING_FILE + ' has been written'

        with open(self.config.VALIDATION_FILE,'wb') as validation_file:
            writer = csv.writer(validation_file, delimiter=';')
            for idx in validation_patches:
                writer.writerow([idx[0], idx[1]])

        print self.config.VALIDATION_FILE + ' has been written'
        print

    def _compute_average(self, DS_DIR):

        with open(self.config.TRAINING_FILE,'r') as training_file:

            pairs = training_file.read().splitlines()
            N = len(pairs)

            print
            print 'Calculating average RGB image and value in training set'
            print '-------------------------------------------------------'
            print 'Number of elements in training set %d'%N

            mean_image = None
            mean_all = np.array([0, 0, 0])

            for p in pairs:
                source, target = p.split(';')
                im = np.array(Image.open(self.config.PATCH_DIR + '/' + source))

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

            Image.fromarray(mean_image,'RGB').save(self.config.TRAINING_AVERAGE_IMAGE)
            print 'Mean image saved to %s'%self.config.TRAINING_AVERAGE_IMAGE


            print
            print
            mean_all = np.array(mean_all)
            mean_all = np.rint(mean_all/float(N)).astype(int)
            print 'Average Value: %s'%mean_all
            print '-----------------------------'

            with open(self.config.TRAINING_AVERAGE_VALUE,'w') as avg_file:
                avg_file.write('%s'%mean_all)
            print 'Average value saved to %s'%self.config.TRAINING_AVERAGE_VALUE


