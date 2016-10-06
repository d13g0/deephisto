"""
Allows creating a network definition programatically.
"""
import os, sys, pdb

os.environ['GLOG_minloglevel'] = '3'

import matplotlib.pylab as plt
from PIL import Image
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
from caffe.coord_map import crop, conv_params

from deephisto import PatchSampler, Console
from CaffeLocations import CaffeLocations

sys.path.insert(0, CaffeLocations.CAFFE_CODE_DIR)


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom,
                         kernel_size=ks,
                         stride=stride,
                         num_output=nout,
                         pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant')
                         )
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


class NetBuilderDeepJet_pool2:
    def __init__(self, name, wsize, data_dir=None):
        """
        Creates the prototxt files associated to a caffe network

        :param name:  Name for the network to be build. A subdirectory under CaffeLocations.NET_DIR is created with this name.
        :param wsize:   width = height dimension of the expected input data for this network
        :param data_dir: location under CaffeLocations.PATCHES_DIR that contains the data that DataLayer will receive during training/testing
        """

        self.name = name
        self.WSIZE = wsize
        self.NET_DIR = CaffeLocations.NET_DIR + '/' + name
        self.SPLIT_DIR = CaffeLocations.SPLIT_DIR + '/' + data_dir

        if data_dir is None:
            self.DATA_DIR = CaffeLocations.PATCHES_DIR
        else:
            self.DATA_DIR = CaffeLocations.PATCHES_DIR + '/' + data_dir

        print
        print 'Network Builder'
        print '----------------'

    def define_structure(self, stage):

        n = caffe.NetSpec()

        if stage != CaffeLocations.STAGE_DEPLOY:
            source_params = dict(stage=stage)
            source_params['data_dir'] = self.DATA_DIR
            source_params['split_dir'] = self.SPLIT_DIR
            source_params['data_augmentation'] = False
            n.data, n.label = L.Python(module='DataLayer',
                                       layer='DataLayer',
                                       ntop=2,
                                       param_str=str(source_params))
        else:
            n.data = L.Input(shape=dict(dim=[1, 3, self.WSIZE, self.WSIZE]))

        # the base net
        n.conv1_1, n.relu1_1 = conv_relu(n.data, 32, pad=7)
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
                                   num_output=CaffeLocations.NUM_LABELS,
                                   kernel_size=1,
                                   pad=0,
                                   param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                   weight_filler=dict(type='xavier'),
                                   bias_filler=dict(type='constant')
                                   )  # must be 1 x num_classes x 1 x 1

        n.upscore_a = L.Deconvolution(n.score_fr,
                                      convolution_param=dict(
                                              num_output=CaffeLocations.NUM_LABELS,
                                              kernel_size=4,
                                              stride=8,
                                              bias_term=False,
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant')
                                      ),
                                      param=[dict(lr_mult=1, decay_mult=1)]
                                      )

        n.score_pool2 = L.Convolution(n.pool2, num_output=CaffeLocations.NUM_LABELS,
                                      kernel_size=1,
                                      pad=0,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant')
                                      )

        n.score_pool2c = crop(n.score_pool2, n.upscore_a)

        n.fuse_pool2 = L.Eltwise(n.upscore_a, n.score_pool2c, operation=P.Eltwise.SUM)

        n.upscore_b = L.Deconvolution(n.fuse_pool2,
                                      convolution_param=dict(num_output=CaffeLocations.NUM_LABELS,
                                                             kernel_size=40,
                                                             stride=4,
                                                             bias_term=False),
                                      param=[dict(lr_mult=1, decay_mult=1)]
                                      )

        # keep = True
        # ks = 40
        # st = ks
        # while(keep):
        #     try:
        #         n.upscore_b = L.Deconvolution(n.fuse_pool2,
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

        n.score = crop(n.upscore_b, n.data)

        if stage != CaffeLocations.STAGE_DEPLOY:
            n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False))

            # n.loss = L.Python(n.score, n.label, module='LossLayer', layer='TopoLossLayer', loss_weight=1)

        return n.to_proto()

    def make(self):

        print
        print 'Making network     : ' + Console.BOLD + Console.OKBLUE + self.name + Console.ENDC
        print 'Patch size         : %d' % self.WSIZE
        print
        print 'Data dir           : %s' % self.DATA_DIR
        print 'Split dir          : %s' % self.SPLIT_DIR

        train_file = self.NET_DIR + '/' + CaffeLocations.TRAINING_PROTO
        val_file = self.NET_DIR + '/' + CaffeLocations.VALIDATION_PROTO
        deploy_file = self.NET_DIR + '/' + CaffeLocations.DEPLOY_PROTO

        train_proto = self.define_structure(CaffeLocations.STAGE_TRAIN)
        val_proto = self.define_structure(CaffeLocations.STAGE_VALIDATION)
        deploy_proto = self.define_structure(CaffeLocations.STAGE_DEPLOY)

        if not os.path.exists(os.path.dirname(train_file)):
            print Console.WARNING + 'Creating directory %s' % os.path.dirname(train_file) + Console.ENDC
            os.makedirs(os.path.dirname(train_file))

        print
        print 'Writing files'
        print
        with open(train_file, 'w') as f:
            f.write(str(train_proto))
            print 'Training network   : %s' % train_file

        with open(val_file, 'w') as f:
            f.write(str(val_proto))
            print 'Validation network : %s' % val_file

        with open(deploy_file, 'w') as f:
            f.write(str(deploy_proto))
            print 'Deploy network     : %s' % deploy_file

        print
        print 'All files have been saved.'

        self.show_structure(train_file)
        self.draw(train_file)

    def show_structure(self, NETWORK_DEFINITION_FILE, SNAPSHOT=None, show_params=False):

        if SNAPSHOT is not None:
            net = caffe.Net(NETWORK_DEFINITION_FILE, caffe.TEST, weights=SNAPSHOT)
        else:
            net = caffe.Net(NETWORK_DEFINITION_FILE, caffe.TRAIN)

        print
        print 'Network Structure:'
        print '------------------'
        print
        print 'Layers'
        print
        for layer_name, blob in net.blobs.iteritems():
            print layer_name.ljust(20) + '\t' + str(blob.data.shape).ljust(15)

        if show_params:
            print
            print 'Parameters (weights) (biases)'
            print
            for layer_name, param in net.params.iteritems():

                try:
                    print layer_name.ljust(20) + '\t' + str(param[0].data.shape).ljust(15), str(param[1].data.shape)
                except IndexError:
                    pass
            print
            print

    def draw(self, fname, orientation='LR'):  # orientation TB, LR, BT


        net = caffe_pb2.NetParameter()
        text_format.Merge(open(fname).read(), net)

        filename = os.path.dirname(fname) + '/%s_network.png' % os.path.splitext(os.path.basename(fname))[0]
        drawing = caffe.draw.draw_net_to_file(net, filename, orientation)

        print 'Network drawn at %s' % filename
        # im = Image.open(filename)
        # plt.imshow(im)
        # plt.show()

