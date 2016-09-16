# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:04:22 2016

@author: dcantor
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
    
    def __init__(self, locations):
        self.locations = locations
        self.configure()
        self.subject = None
        

    def configure(self):
        """
        Configure the paths for the ImageRetriever object to operate
        Change the remote directories according to your mount point (keep the wildcard %s)
        """
        root = self.locations.ROOT_DIR

        local_exvivo_dir = root + '/subjects/%s/exvivo/'
        local_histo_dir = root + '/subjects/%s/hist/'

        remote_exvivo_dir = '/home/dcantor/epilepsy/%s/Processed/Ex-Hist_Reg/9.4T/Neo/aligned_Ex_100um/'
        remote_histo_dir = '/home/dcantor/histology/Histology/%s/100um_5umPad_FeatureMaps/aligned/Neo_NEUN/'

        self.source = {
            'FA': remote_exvivo_dir + 'dti_smoothed_0.2/dti_FA.100um.nii.gz',
            'MD': remote_exvivo_dir + 'dti_smoothed_0.2/dti_MD.100um.nii.gz',
            'MR': remote_exvivo_dir + 'reg_ex_mri_100um.nii.gz',
            'HI': remote_histo_dir  + 'count_deformable_100um.nii.gz'
        }
        

        self.target = {
            'FA': local_exvivo_dir + 'dti_FA.100um.nii.gz',
            'MD': local_exvivo_dir + 'dti_MD.100um.nii.gz',
            'MR': local_exvivo_dir + 'reg_ex_mri_100um.nii.gz',
            'HI': local_histo_dir  + 'count_deformable_100um.nii.gz'
            
        }
        
    def set_subject(self, subject):
        self.subject = subject
        self.configure()
        print 'ImageRetriever: looking at subject %s'%subject
        for key,item in self.source.iteritems():
            self.source[key] = item%subject
              
        for key, item in self.target.iteritems():
            self.target[key] = item%subject
        
        
    def inspect_subject(self, subject):
        """
        Determines if all the required files exist in the database for a given subject
        """
        
        print '--------------------------------------------------------------'
        print ' Inspecting subject %s' % Console.BOLD + subject + Console.ENDC
        print '--------------------------------------------------------------'
        
        self.set_subject(subject)
        
        subject_ok = True
        
        for key,item in self.source.iteritems():
    
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
        
        

    def retrieve(self,subject):
        if subject is None:
            subject = self.subject

        if not self.inspect_subject(subject):
            print 'Something is missing. Data for this subject cannot be retrieved'
            return
   
     
        
        for key in self.target.keys():
            origen  = self.source[key]
            destino = self.target[key]
            if not os.path.exists(os.path.dirname(destino)):
                try:
                    os.makedirs(os.path.dirname(destino))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        print exc
            
            print 'Copying %s'%(key)
            copyfile(origen,destino)
            
        print 'Files have been retrieved'
