import os
os.environ['GLOG_minloglevel'] = '3'
from deephisto import NetBuilder

def dh_create_network(net_name, wsize, data_dir):
    nd = NetBuilder(net_name, wsize, data_dir=data_dir)
    nd.make()


if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    dh_create_network('dh28b', 28, '28x28b')