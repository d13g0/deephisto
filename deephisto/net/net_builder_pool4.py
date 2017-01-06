#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor


import os, sys, pdb
os.environ['GLOG_minloglevel'] = '3'
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

from net_definitions import NetStage, conv_relu, max_pool, make_net

# adds the directory where this script resides so net can load the DataLayer
# @see: https://github.com/rbgirshick/py-faster-rcnn/issues/98
sys.path.insert(0,os.path.dirname(__file__))


class NetBuilder_pool4:

    def __init__(self, config):
        self.NAME = config.NETWORK_NAME
        self.NET_DIR = config.NETWORK_DIR
        self.DATASET_DIR = config.DATASET_DIR
        self.PATCH_DIR = config.PATCH_DIR
        self.WSIZE = config.PATCH_SIZE
        self.NUM_LABELS = config.NUM_LABELS
        self.TRAINING_PROTO = config.TRAINING_PROTO
        self.VALIDATION_PROTO = config.VALIDATION_PROTO
        self.DEPLOY_PROTO = config.DEPLOY_PROTO
        self.AVERAGE_IMAGE = config.TRAINING_AVERAGE_IMAGE
        self.TRAINING_PATCHES = config.TRAINING_PATCHES
        self.VALIDATION_PATCHES = config.VALIDATION_PATCHES
        self.RANDOM_TRAINING = config.RANDOM_TRAINING
        self.INCLUDE_ROTATIONS = config.INCLUDE_ROTATIONS

        print
        print 'Network Builder POOL4'
        print '---------------------'

    def define_structure(self, stage):

        n = caffe.NetSpec()

        if stage != NetStage.DEPLOY:
            source_params = dict(stage=stage)
            source_params['dataset_dir'] = self.DATASET_DIR
            source_params['patch_dir'] = self.PATCH_DIR
            source_params['average_image'] = self.AVERAGE_IMAGE
            source_params['training_patches'] = self.TRAINING_PATCHES
            source_params['validation_patches'] = self.VALIDATION_PATCHES
            source_params['random_training'] = self.RANDOM_TRAINING
            source_params['include_rotations'] = self.INCLUDE_ROTATIONS

            n.data, n.label = L.Python(module='DataLayer', layer='DataLayer', ntop=2, param_str=str(source_params))
        else:
            n.data = L.Input(shape=dict(dim=[1, 3, self.WSIZE, self.WSIZE]))

        # the base net
        n.conv1_1, n.relu1_1 = conv_relu(n.data, 32, pad=12)
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 32)
        n.pool1 = max_pool(n.conv1_2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 64)
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 64)
        n.pool2 = max_pool(n.relu2_2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 128)
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 128)
        n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 128)
        n.pool3 = max_pool(n.relu3_3)

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 256)
        n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 256)
        n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 256)
        n.pool4 = max_pool(n.relu4_3)

        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 256)
        n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 256)
        n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 256)
        n.pool5 = max_pool(n.relu5_3)

        # fully conv
        n.fc6, n.relu6 = conv_relu(n.pool5, 2048, ks=2, pad=0)
        n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

        n.fc7, n.relu7 = conv_relu(n.drop6, 2048, ks=1, pad=0)
        n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

        n.score_fr = L.Convolution(n.drop7,
                                   num_output=self.NUM_LABELS,
                                   kernel_size=1,
                                   pad=0,
                                   param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                   weight_filler=dict(type='xavier'),
                                   bias_filler=dict(type='constant')
                                   )  # must be 1 x num_classes x 1 x 1
        # ks = 4
        # st = 8
        # while(True):

        n.upscore_a = L.Deconvolution(n.score_fr,
                                      convolution_param=dict(
                                              num_output=self.NUM_LABELS,
                                              kernel_size=4,
                                              stride=2,
                                              bias_term=False,
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant')
                                      ),
                                      param=[dict(lr_mult=1, decay_mult=1)]
                                      )

        n.score_pool4 = L.Convolution(n.pool4, num_output=self.NUM_LABELS,
                                      kernel_size=1,
                                      pad=0,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant')
                                      )
        #print ks, st

        #    try:
        n.score_pool4c = crop(n.score_pool4, n.upscore_a)
        #        break
        #    except AssertionError:
        #        st = st - 1

        #print
        #print 'FOUND'
        #print ks, st
        #raw_input()

        n.fuse_pool4 = L.Eltwise(n.upscore_a, n.score_pool4c, operation=P.Eltwise.SUM)

        n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
                                      convolution_param=dict(num_output=self.NUM_LABELS,
                                                             kernel_size=32,
                                                             stride=16,
                                                             bias_term=False),
                                      param=[dict(lr_mult=1, decay_mult=1)]
                                      )


        # ks = 32
        # st = ks
        # while(True):
        #     try:
        #         n.upscore_b = L.Deconvolution(n.fuse_pool4,
        #                                       convolution_param=dict(num_output=CaffeLocations.NUM_LABELS,
        #                                                              kernel_size=ks,
        #                                                              stride=st,
        #                                                              bias_term=False),
        #                                       param=[dict(lr_mult=0)])
        #
        #         n.score = crop(n.upscore_b, n.data)
        #         break
        #     except AssertionError as e:
        #         st=st-1
        #
        # print
        # print 'FOUND'
        # print ks, st

        n.score = crop(n.upscore_pool4, n.data)

        if stage != NetStage.DEPLOY:
            n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False))
        #else:
            #n.output = L.Softmax(n.score)
            #n.loss = L.Python(n.score, n.label, module='LossLayer', layer='TopoLossLayer', loss_weight=1)

        return n.to_proto()

    def make(self):
        make_net(self)
