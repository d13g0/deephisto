class CaffeLocations:
    """
    This file contains variables and paths specific to caffe processing.
    """
    NUM_LABELS = 10

    # TRAINING_MEAN = np.array([136, 99, 99], dtype=np.float32)

    DH_ROOT = '/home/dcantor/projects/deephisto'

    PATCHES_DIR = DH_ROOT + '/patches'
    CODE_DIR = DH_ROOT + '/code'
    CAFFE_WORKDIR = DH_ROOT + '/caffe'

    DATA_DIR = CAFFE_WORKDIR + '/data'
    SNAPSHOT_DIR = DATA_DIR + '/%s/_iter_%d.caffemodel'

    SPLIT_DIR = CAFFE_WORKDIR + '/split'
    AVG_IMG = SPLIT_DIR + '/training_avg.png'

    NET_DIR = CAFFE_WORKDIR + '/net'
    DEPLOY_NET_PROTOTXT = NET_DIR + '/deploy.prototxt'

    TRAINING_PROTO = 'train.prototxt'
    VALIDATION_PROTO = 'val.prototxt'

    STAGE_TRAIN = 'training'
    STAGE_VALIDATION = 'validation'

    CAFFE_CODE_DIR = CODE_DIR + '/caffe'
