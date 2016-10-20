import os, pdb
os.environ['GLOG_minloglevel'] = '3'
import caffe
from deephisto import Locations, NetBuilder, NetBuilderDeepJet_pool4, NetBuilderDeepJet_pool2, NetBuilderDeepJet_fcn8


def dh_create_network(net_name, wsize, data_dir):
    nd = NetBuilder(net_name, wsize, data_dir=data_dir)
    nd.make()

def dh_create_deepjet_2():
    nd = NetBuilderDeepJet_pool2('dhjet_2', 28, data_dir='28x28b')
    nd.make()


def dh_create_deepjet_4():
    nd = NetBuilderDeepJet_pool4('dhjet_4', 28, data_dir='28x28b')
    nd.make()

def dh_create_deepjet_fcn8():
    nd = NetBuilderDeepJet_fcn8('fcn8e', 28, data_dir='28x28e')
    nd.make()

if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    #dh_create_network('dhloss', 28, '28x28b')
    #dh_create_deepjet_2()
    #dh_create_deepjet_4()
    dh_create_deepjet_fcn8()
