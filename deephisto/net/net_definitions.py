#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor
import os
from google.protobuf import text_format
from caffe import layers as L, params as P
import caffe.draw
from caffe.proto import caffe_pb2

from deephisto import Console

class NetStage:
    TRAINING = 'training'
    VALIDATION = 'validation'
    DEPLOY     = 'deploy'



def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    """
    Defines a convolutional layer using net API
    """
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
    """
    Defines a pooling layer using net API
    """
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def draw_net(fname, orientation='LR'):
    """
    Creates a PNG file for the given caffe network
    :param fname: name of the network
    :param orientation: LR left-right BT: bottom-top TB: top-bottom
    :return:
    """
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(fname).read(), net)
    filename = os.path.dirname(fname) + '/%s_network.png' % os.path.splitext(os.path.basename(fname))[0]
    drawing = caffe.draw.draw_net_to_file(net, filename, orientation)

    print 'Network drawn at %s' % filename

    # import matplotlib.pylab as plt
    # from PIL import Image
    # im = Image.open(filename)
    # plt.imshow(im)
    # plt.show()


def show_structure(protofile, snapshot=None, show_params=False):
    if snapshot is not None:
        net = caffe.Net(protofile, caffe.TEST, weights=snapshot)
    else:
        net = caffe.Net(protofile, caffe.TRAIN)

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


def make_net(netobj):
    print
    print 'Making network     : ' + Console.BOLD + Console.OKBLUE + netobj.NAME + Console.ENDC
    print 'Patch size         : %d' % netobj.WSIZE
    print
    print 'Dataset directory   : %s' % netobj.DATASET_DIR
    print 'Patch directory     : %s' % netobj.PATCH_DIR

    train_proto = netobj.define_structure(NetStage.TRAINING)
    val_proto = netobj.define_structure(NetStage.VALIDATION)
    deploy_proto = netobj.define_structure(NetStage.DEPLOY)

    if not os.path.exists(netobj.NET_DIR):
        print Console.WARNING + 'Creating directory %s' % netobj.NET_DIR + Console.ENDC
        os.makedirs(netobj.NET_DIR)

    print
    print 'Writing files'
    print
    with open(netobj.TRAINING_PROTO, 'w') as f:
        f.write(str(train_proto))
        print 'Training network   : %s' % netobj.TRAINING_PROTO

    with open(netobj.VALIDATION_PROTO, 'w') as f:
        f.write(str(val_proto))
        print 'Validation network : %s' % netobj.VALIDATION_PROTO

    with open(netobj.DEPLOY_PROTO, 'w') as f:
        f.write(str(deploy_proto))
        print 'Deploy network     : %s' % netobj.DEPLOY_PROTO

    print
    print 'All files have been saved.'

    show_structure(netobj.TRAINING_PROTO)
    draw_net(netobj.TRAINING_PROTO)