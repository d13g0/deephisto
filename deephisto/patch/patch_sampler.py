import random

import numpy as np

from deephisto.utils.console import Console


class PatchSampler:
    """
    Samples and extract regions from a given binary mask (bmask)

    This class uses numpy, where arrays are row-major. To avoid confusion
    between image coordinates and cartesian coordinates, this class follows
    the convention row, col instead of using the traditional x,y variables:

              cols
        ----------------> x (w)
        |
  rows  |
        |
        v y (h)
        In numpy first dimension is always rows!

        y,x  = row, col


    """
    WSIZE = 30
    WSIDE = WSIZE / 2
    S_MONTECARLO = 'Montecarlo'
    S_SYSTEMATIC = 'Systematic'
    S_OVERLAP = 'Overlap'

    def __init__(self, bmask):

        self.set_mask(bmask)
        self.samplers = {
            PatchSampler.S_MONTECARLO: self.montecarlo_sampling,
            PatchSampler.S_SYSTEMATIC: self.systematic_sampling,
            PatchSampler.S_OVERLAP   : self.overlap_sampling,
        }

    def set_mask(self, bmask):
        self.bmask = bmask
        (rows, cols) = np.where(self.bmask > 0)  # list of coordinates
        self.coords = zip(rows, cols)
        self.L = len(self.coords)

    def check_convex(self, row, col):

        r1 = row - PatchSampler.WSIDE
        r2 = row + PatchSampler.WSIDE
        c1 = col - PatchSampler.WSIDE
        c2 = col + PatchSampler.WSIDE

        coords = self.coords
        return (r1, c1) in coords and (r2, c2) in coords

    def sample(self, type, params=None, callback=None):
        """
        :param type: type of sampling
        :param callback: function to callback when a match is found (need to receive x,y coordinates)
        """
        sampler = self.samplers.get(type, self.overlap_sampling)

        print '\n\nInitiating sampling'
        print '-------------------------------'
        print 'Type               : ' + Console.BOLD + type + Console.ENDC
        print 'Candidate list size: %d' % self.L

        return sampler(params, callback)

    def montecarlo_sampling(self, params, callback):

        if params is not None and ('C' in params):
            C = params['C']
        else:
            C = 50

        print 'Coverage           : %d%%' % C
        N = int(self.L * C / 100)
        print 'Number of samples  : ' + Console.OKBLUE + '%d' % N + Console.ENDC

        idx = random.sample(range(self.L), N)  # candidate indices
        selected = []

        print 'One moment please ....'
        for i in idx:
            (row, col) = self.coords[i]
            if self.check_convex(row, col):
                selected.append((row, col))

        print 'Number of selections: ' + Console.OKBLUE + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def systematic_sampling(self, params, callback):
        """
        Finds windows of a given size that fit in the mask, and returns the center pixel for each
        one of those windows.
        """
        selected = []
        for (row, col) in self.coords:
            if self.check_convex(row, col):
                selected.append((row, col))
                if callback is not None: callback(row, col)

        return selected

    def overlap_sampling(self, params, callback):

        selected = []

        row, col = self.coords[0]  # numpy first dimension is rows
        height, width = self.bmask.shape

        STEP = PatchSampler.WSIZE / 5

        print 'One moment please ....'

        while (row <= height and col <= width):
            while (col < width):

                if (row, col) in self.coords:
                    selected.append((row, col))
                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected
