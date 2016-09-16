"""
Allows creating a network definition programatically.
"""
import os

os.environ['GLOG_minloglevel'] = '2'
import caffe
import caffe.draw
from caffe.proto import caffe_pb2
from caffe import layers as L, params as P
from caffe.coord_map import crop
from google.protobuf import text_format

from CaffeLocations import CaffeLocations


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


class NetworkDescriptor:
    def __init__(self, descriptor_dir=CaffeLocations.NET_DIR,
                 data_dir=CaffeLocations.PATCHES_DIR):

        self.descriptor_dir = descriptor_dir
        self.data_dir = data_dir

        print
        print 'Network descriptor initialized'
        print '--------------------------------'

    def define_structure(self, stage):

        n = caffe.NetSpec()
        source_params = dict(stage=stage)  # ,  seed=1337)
        source_params['data_dir'] = self.data_dir
        source_params['data_augmentation'] = False

        n.data, n.label = L.Python(module='DataLayer', layer='DataLayer', ntop=2, param_str=str(source_params))

        # the base net
        n.conv1_1, n.relu1_1 = conv_relu(n.data, 8, pad=16)
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 8)
        n.pool1 = max_pool(n.conv1_2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 16)
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 16)
        n.pool2 = max_pool(n.relu2_2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 32)
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 32)
        n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 32)
        n.pool3 = max_pool(n.relu3_3)

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 64)
        n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 64)
        n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 64)
        n.pool4 = max_pool(n.relu4_3)

        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 128)
        n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 128)
        n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 128)
        n.pool5 = max_pool(n.relu5_3)

        # fully conv
        n.fc6, n.relu6 = conv_relu(n.pool5, 512, ks=2, pad=0)
        # n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

        n.fc7, n.relu7 = conv_relu(n.relu6, 512, ks=1, pad=0)
        # n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

        n.score_fr = L.Convolution(n.relu7,
                                   num_output=CaffeLocations.NUM_LABELS,
                                   kernel_size=1,
                                   pad=0,
                                   param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                   weight_filler=dict(type='xavier'),
                                   bias_filler=dict(type='constant')
                                   )  # must be 1 x num_classes x 1 x 1

        n.deconv = L.Deconvolution(n.score_fr,
                                   convolution_param=dict(
                                           num_output=CaffeLocations.NUM_LABELS,
                                           kernel_size=64,
                                           stride=32,
                                           bias_term=False,
                                           weight_filler=dict(type='xavier'),
                                           bias_filler=dict(type='constant')
                                   ),
                                   # param=[dict(lr_mult=0)], #do not learn this filter?
                                   param=[dict(lr_mult=1, decay_mult=1)]
                                   )

        n.score = crop(n.deconv, n.data)
        n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False))

        return n.to_proto()

    def make(self):

        train_file = self.descriptor_dir + '/' + CaffeLocations.TRAINING_PROTO
        val_file = self.descriptor_dir + '/' + CaffeLocations.VALIDATION_PROTO

        with open(train_file, 'w') as f:
            f.write(str(self.define_structure(CaffeLocations.STAGE_TRAIN)))
            print '%s generated' % train_file

        with open(val_file, 'w') as f:
            f.write(str(self.define_structure(CaffeLocations.STAGE_VALIDATION)))
            print '%s generated' % val_file

        self.show_structure(train_file)

        print 'DONE'

    def show_structure(self, NETWORK_DEFINITION_FILE, SNAPSHOT=None):

        if SNAPSHOT is not None:
            net = caffe.Net(NETWORK_DEFINITION_FILE, caffe.TEST, weights=SNAPSHOT)
        else:
            net = caffe.Net(NETWORK_DEFINITION_FILE, caffe.TRAIN)

        print '\n\nNETWORK STRUCTURE'
        print '----------------------------------------------------------------'
        print
        print 'Layers'
        print
        for layer_name, blob in net.blobs.iteritems():
            print layer_name.ljust(20) + '\t' + str(blob.data.shape).ljust(15)
        print
        print 'Parameter Shapes (weights) (biases)'
        print
        for layer_name, param in net.params.iteritems():

            try:
                print layer_name.ljust(20) + '\t' + str(param[0].data.shape).ljust(15), str(param[1].data.shape)
            except IndexError:
                pass
        print
        print

    def draw(self, fname, orientation='TB'):  # orientation TB, LR, BT

        fname = CaffeLocations.CAFFE_WORKDIR + '/%s' % fname
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(fname).read(), net)

        import matplotlib.pylab as plt
        from PIL import Image
        filename = os.path.dirname(fname) + '/%s.png' % os.path.splitext(os.path.basename(fname))[0]
        drawing = caffe.draw.draw_net_to_file(net, filename, orientation)
        im = Image.open(filename)
        plt.imshow(im)
        plt.show()


if __name__ == '__main__':
    nd = NetworkDescriptor()
    nd.make()
    # nd.draw('train.prototxt','BT')
