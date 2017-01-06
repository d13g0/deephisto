#  This file makes part of DEEP HISTO
#
#  Deep Histo is a medical imaging project that uses deep learning to
#  predict histological features from MRI.
#
#  Author: Diego Cantor

import pdb, re, argparse
from ConfigParser import SafeConfigParser

class ObjectDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class Config:
    REMOTE_FILES_KEY = 'remote files'
    LOCAL_FILES_KEY  ='local files'
    VISUALIZER       = 'visualizer'

    def __init__(self,d):
        self.data = d
        self.remote_files = ObjectDict({
            'FA': d[Config.REMOTE_FILES_KEY]['fa'],
            'MD': d[Config.REMOTE_FILES_KEY]['md'],
            'MR': d[Config.REMOTE_FILES_KEY]['mr'],
            'HI': d[Config.REMOTE_FILES_KEY]['hi'],
        })

        self.local_files = ObjectDict({
            'FA': d[Config.LOCAL_FILES_KEY]['fa'],
            'MD': d[Config.LOCAL_FILES_KEY]['md'],
            'MR': d[Config.LOCAL_FILES_KEY]['mr'],
            'HI': d[Config.LOCAL_FILES_KEY]['hi'],
        })

        self.visualizer = ObjectDict({
            'WSIZE' : int(d[Config.VISUALIZER]['wsize']),
            'COLORMAP': d[Config.VISUALIZER]['colormap']
        })

    @property
    def STUDY_NAME(self):
        return self.data['study']['name']

    @property
    def SUBJECTS(self):
        return map(int, self.data['study']['subjects'].split(','))

    @property
    def ROOT(self):
        return self.data['root']['root']

    @property
    def HISTO_PNG(self):
        return self.data['png']['histo_png']

    @property
    def LABELS_PNG(self):
        return self.data['png']['labels_png']

    @property
    def SOURCE_PNG(self):
        return self.data['png']['source_png']

    @property
    def IMAGES_DIR(self):
        return self.data['png']['images_dir']

    @property
    def MASK_DIR(self):
        return self.data['png']['mask_dir']

    @property
    def SLICE_MASK_PNG(self):
        return self.data['png']['slice_mask_png']

    @property
    def ANNOTATIONS_ZIP(self):
        return self.data['png']['annotations_zip']

    @property
    def NUM_LABELS(self):
        return int(self.data['classification']['num_labels'])

    @property
    def GAUSSIAN_BLUR(self):
        return self.data['classification']['gaussian_blur'] in  ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

    @property
    def GAUSSIAN_RADIUS(self):
        return float(self.data['classification']['gaussian_radius'])

    @property
    def HISTOLOGY_RANGE_MIN(self):
        return float(self.data['histology']['range_min'])

    @property
    def HISTOLOGY_RANGE_MAX(self):
        return float(self.data['histology']['range_max'])

    @property
    def PATCH_SIZE(self):
        return int(self.data['patches']['patch_size'])

    @property
    def PATCH_DIR(self):
        return self.data['patches']['patch_dir']

    @property
    def PATCH_TEMPLATE(self):
        return self.data['patches']['patch_template']

    @property
    def TRAINING_PERCENTAGE(self):
        return float(self.data['classification']['training_percentage'])

    @property
    def DATASET_DIR(self):
        return self.data['classification']['dataset_dir']

    @property
    def TRAINING_PATCHES(self):
        return self.data['classification']['training_patches']

    @property
    def VALIDATION_PATCHES(self):
        return self.data['classification']['validation_patches']

    @property
    def TRAINING_AVERAGE_IMAGE(self):
        return self.data['classification']['training_avg_image']

    @property
    def TRAINING_AVERAGE_VALUE(self):
        return self.data['classification']['training_avg_value']

    @property
    def NETWORK_NAME(self):
        return self.data['network']['name']

    @property
    def NETWORK_DIR(self):
        return self.data['network']['network_dir']

    @property
    def TRAINING_PROTO(self):
        return self.data['network']['training_proto']
    @property
    def VALIDATION_PROTO(self):
        return self.data['network']['validation_proto']

    @property
    def DEPLOY_PROTO(self):
        return self.data['network']['deploy_proto']

    @property
    def RANDOM_TRAINING(self):
        return self.data['network']['random_training'] in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup',
                                                                 'certainly', 'uh-huh']

    @property
    def INCLUDE_ROTATIONS(self):
        return self.data['network']['include_rotations'] in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup',
                                                                 'certainly', 'uh-huh']




def _check_for_variables(parser):
    """
    Checks if there are unresolved variables
    """
    for section in parser.sections():
        for key,value in parser.items(section):
            if len(re.findall(r"\$", value))>0:
                #print value
                return True
    return False


def _search_and_replace(parser, section):
    """
    Replaces references to other configuration variables in the same file
    """
    INTERPOLATION_RE = re.compile(r"\$\{(?:(?P<section>[^:]+):)?(?P<key>[^}]+)\}")
    result = []
    def interpolate_func(match):
        d = match.groupdict()
        s = d.get('section')
        if s is None:
            s = section
        key = d.get('key')
        return parser.get(s, key)

    for key, value in parser.items(section):
        value = re.sub(INTERPOLATION_RE, interpolate_func, value)
        result.append(
            (key,value)
        )
    return result


def _setup_config_object(parser):
    data = dict()
    data.update(parser._sections)
    config = Config(data)
    return config


def dh_read_config(filename):
    """
    Reads the configuration file and returns a config object (dictionary)
    """
    parser = SafeConfigParser()
    c = parser.read(filename)
    if len(c)==0:
        raise ValueError('The configuration file %s is unreadable'%filename)

    def iteration():
        for section in parser.sections():
            results = _search_and_replace(parser, section)
            for item in results:
                key,value = item
                parser.set(section,key,value)
            print


    iteration()
    count = 0
    max_depth = 10
    while _check_for_variables(parser):
        iteration()
        count = count  + 1
        if count == max_depth:
            raise Exception('There are undefined variables in the configuration file [%s]'%filename)


    config = _setup_config_object(parser)
    print 'Study %s'%config.STUDY_NAME
    print '-------------------------------'
    print
    return config

def dh_load_subjects(config):
    subjects = []
    for i in config.SUBJECTS:
        subjects.append('EPI_P%03d' % i)
    return subjects


def dh_config_selector():
    parser = argparse.ArgumentParser(description='Select a configuration file')
    parser.add_argument('-c', action='store', dest='configuration_file', required=True)
    results=parser.parse_args()
    return dh_read_config(results.configuration_file)
