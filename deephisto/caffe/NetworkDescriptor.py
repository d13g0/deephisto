"""
Allows creating a network definition programatically.
"""

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop



def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant')
                         )
    return conv, L.ReLU(conv, in_place=True)

def relu(bottom):
    return L.ReLU(bottom)

def conv(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom,
                         kernel_size=ks,
                         stride=stride,
                         num_output=nout,
                         pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant')
                         )
    return conv

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

class NetworkDescriptor:


    def __init__(self):
        print
        print 'Network descriptor initialized'
        print '--------------------------------'

    def define_structure(self, stage, data_dir):

        n = caffe.NetSpec()
        pydata_params = dict(stage=stage,  seed=1337)
        pydata_params['data_dir'] = data_dir

        n.data, n.label = L.Python(module='PatchSourceLayer', layer='PatchSourceLayer', ntop=2, param_str=str(pydata_params))

        # the base net
        n.conv1_1, n.relu1_1 = conv_relu(n.data, 8,pad=50)
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 8)
        n.pool1 = max_pool(n.conv1_2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 16)
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 16)
        n.pool2 = max_pool(n.relu2_2)


        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 32)
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 32)
        n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 32)
        n.pool3 = max_pool(n.relu3_3)

        # n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
        # n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
        # n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
        # n.pool4 = max_pool(n.relu4_3)

        # n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
        # n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
        # n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
        # n.pool5 = max_pool(n.relu5_3)

        # fully conv
        n.fc6, n.relu6 = conv_relu(n.pool3, 64, ks=7, pad=0)
        #n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

        n.fc7, n.relu7 = conv_relu(n.relu6, 64, ks=1, pad=0)
        #n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

        n.score_fr = L.Convolution(n.relu7,
                                   num_output=20,
                                   kernel_size=1,
                                   pad=0,
                                   param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=0)],
                                   weight_filler=dict(type='xavier'),
                                   bias_filler=dict(type='constant')
                                   )

        n.deconv = L.Deconvolution(n.score_fr,
                                   convolution_param=dict(
                                       num_output=20,
                                       kernel_size=16,
                                       stride=8,
                                       bias_term=False ,
                                       weight_filler=dict(type='xavier'),
                                       bias_filler=dict(type='constant')
                                   ),
                                   param=[dict(lr_mult=0)],
                                   )

        n.score = crop(n.deconv, n.data)
        n.loss = L.SoftmaxWithLoss(n.score, n.label,loss_param=dict(normalize=False))

        return n.to_proto()

    def make(self, NET_SPEC_DIR, DATA_DIR):
        train_file = NET_SPEC_DIR + '/train.prototxt'
        val_file = NET_SPEC_DIR + '/val.prototxt'

        with open(train_file, 'w') as f:
            f.write(str(self.define_structure('training', DATA_DIR)))
            print '%s generated'%train_file

        with open(val_file, 'w') as f:
            f.write(str(self.define_structure('validation', DATA_DIR)))
            print '%s generated'%val_file

        print 'DONE'

    #def add_kernel_params(self, file):



    # def show_structure(self,definition_dir):
    #     net = caffe.Net(NETWORK_DEFINITION_FILE, TRAINED_MODEL, caffe.TEST)
    #     net_structure(net)

    def show_structure(self, NETWORK_DEFINITION_FILE, SNAPSHOT):
        net = caffe.Net(NETWORK_DEFINITION_FILE, SNAPSHOT, caffe.TEST)

        print '----------------------------------------------------------------'
        print 'NETWORK STRUCTURE'
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
            hasdata = np.any(param[0].data > 0)
            try:
                print layer_name.ljust(20) + '\t' + str(param[0].data.shape).ljust(15), str(param[1].data.shape)
            except IndexError:
                pass
        print
        print


if __name__=='__main__':
    nd = NetworkDescriptor()
    nd.make('/home/dcantor/projects/deephisto/caffe','/home/dcantor/projects/deephisto/patches')