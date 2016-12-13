# -*- coding: utf-8 -*-
"""
@author: Diego Cantor
"""

import errno
import os
from shutil import copyfile

from deephisto.utils.console import Console


class ImageRetriever:
    """
    Copies the NIFTI files from the epilepsy and histology databases to the local disk for faster access
    and processing
    """

    def __init__(self, config):

        self.subject = {}

        self.sources = {  # a map indicating the remote locations ofr these files
            'FA': config.remote_files.FA,
            'MD': config.remote_files.MD,
            'MR': config.remote_files.MR,
            'HI': config.remote_files.HI
        }

        self.targets = {  # a map indicating where the remote files will be copied
            'FA': config.local_files.FA,
            'MD': config.local_files.MD,
            'MR': config.local_files.MR,
            'HI': config.local_files.HI

        }

    def set_subject(self, name):

        self.subject['name'] = name
        self.subject['source'] = {}
        self.subject['target'] = {}

        print 'ImageRetriever: looking at subject %s' % name

        for key, path in self.sources.iteritems():
            self.subject['source'][key] = path.format(subject=name)

        for key, path in self.targets.iteritems():
            self.subject['target'][key] = path.format(subject=name)

    def inspect_subject(self, name):
        """
        Determines if all the required files exist in the database for a given subject
        """
        print
        print '--------------------------------------------------------------'
        print ' Inspecting subject %s' % Console.BOLD + name + Console.ENDC
        print '--------------------------------------------------------------'

        self.set_subject(name)

        subject_ok = True

        for key, item in self.subject['source'].iteritems():

            if os.path.exists(item):
                flag = Console.OKGREEN + 'OK' + Console.ENDC
            else:
                flag = Console.WARNING + 'Not found' + Console.ENDC
                subject_ok = False

            print '%s %s' % (key, flag)

        if subject_ok:
            print 'SUBJECT OK'
        else:
            print Console.FAIL + 'ERROR: MISSING DATA' + Console.ENDC
        return subject_ok

    def retrieve(self, name):

        if not self.inspect_subject(name):
            print 'Something is missing. Data for this subject cannot be retrieved'
            return

        for key in self.subject['target'].keys():
            origen = self.subject['source'][key]
            destino = self.subject['target'][key]
            if not os.path.exists(os.path.dirname(destino)):
                try:
                    os.makedirs(os.path.dirname(destino))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        print exc

            print 'Copying %s' % (key)
            copyfile(origen, destino)

        print 'Files have been retrieved'
