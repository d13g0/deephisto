import numpy as np

class CaffeSettings:
    """
    This file contains variables and paths specific to caffe processing.
    """
    NUM_LABELS = 10

    TRAINING_MEAN = np.array([126, 94, 94], dtype=np.float32)

    DH_ROOT = '/home/dcantor/projects/deephisto'

    PATCHES_DIR = DH_ROOT + '/patches'
    CODE_DIR = DH_ROOT + '/code'
    CAFFE_WORKDIR = DH_ROOT + '/caffe'

    DATA_DIR = CAFFE_WORKDIR + '/data'
    SNAPSHOT_DIR = DATA_DIR + '/%s/_iter_%d.caffemodel'

    SPLIT_DIR = CAFFE_WORKDIR + '/split'
    AVG_IMG =  'training_avg.png'

    NET_DIR = CAFFE_WORKDIR + '/net'

    TRAINING_PROTO   = 'train.prototxt'
    VALIDATION_PROTO = 'val.prototxt'
    DEPLOY_PROTO     = 'deploy.prototxt'

    STAGE_TRAIN = 'training'
    STAGE_VALIDATION = 'validation'
    STAGE_DEPLOY     = 'deploy'

    CAFFE_CODE_DIR = CODE_DIR + '/caffe'
