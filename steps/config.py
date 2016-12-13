import pdb, re
from ConfigParser import SafeConfigParser

class ObjectDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    # def __setattr__(self, name, value):
    #     self[name] = value
    #
    # def __delattr__(self, name):
    #     if name in self:
    #         del self[name]
    #     else:
    #         raise AttributeError("No such attribute: " + name)


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




def check_for_variables(parser):
    """
    Checks if there are unresolved variables
    """
    for section in parser.sections():
        for key,value in parser.items(section):
            if len(re.findall(r"\$", value))>0:
                #print value
                return True
    return False


def search_and_replace(parser, section):
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


def setup_config_object(parser):
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
            results = search_and_replace(parser, section)
            for item in results:
                key,value = item
                parser.set(section,key,value)
            print


    iteration()
    count = 0
    max_depth = 10
    while check_for_variables(parser):
        iteration()
        count = count  + 1
        if count == max_depth:
            raise Exception('There are undefined variables in the configuration file [%s]'%filename)


    config = setup_config_object(parser)
    print 'Study %s'%config.STUDY_NAME
    print '-------------------------------'
    print
    return config

def dh_load_subjects(config):
    subjects = []
    for i in config.SUBJECTS:
        subjects.append('EPI_P%03d' % i)
    return subjects

if __name__ == '__main__':
    CONFIG = dh_read_config('/home/dcantor/projects/deephisto/code/config_neuronal_density.ini')
    raw_input('Configuration file read. Congrats! Press any key to finish')