import os, pdb
os.environ['GLOG_minloglevel'] = '3'
import caffe
from deephisto import Locations, NetBuilder, NetBuilderDeepJet_pool4, NetBuilderDeepJet_pool2


def dh_create_network(net_name, wsize, data_dir):
    nd = NetBuilder(net_name, wsize, data_dir=data_dir)
    nd.make()

def dh_create_deepjet(net_name, wsize, data_dir):
    nd = NetBuilderDeepJet_pool2(net_name, wsize, data_dir=data_dir)
    nd.make()


#

if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    #dh_create_network('dhloss', 28, '28x28b')
    dh_create_deepjet('dhjet',28,'28x28b')
