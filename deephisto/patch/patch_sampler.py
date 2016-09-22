import random

import numpy as np
from deephisto.utils import Console


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

    TYPE_MONTECARLO = 'Montecarlo'
    TYPE_OVERLAP = 'Overlap'
    TYPE_CONVEX = 'Convex'
    TYPE_BACKGROUND = 'Background'
    TYPE_ALL = 'All'

    def __init__(self, wsize=30, type=TYPE_ALL, mask=None, params=None, callback=None):

        self.type = type
        self.params = params
        self.WSIZE = wsize
        self.WSIDE = self.WSIZE / 2
        self.callback = callback

        self.factory = {
            PatchSampler.TYPE_MONTECARLO: self.montecarlo_sampling,
            PatchSampler.TYPE_OVERLAP   : self.overlap_sampling,
            PatchSampler.TYPE_CONVEX    : self.convex_sampling,
            PatchSampler.TYPE_ALL       : self.all_sampling,
            PatchSampler.TYPE_BACKGROUND: self.background_sampling
        }

        if mask == None:
            print Console.HEADER + ' Warning: the mask for the sampler is not set yet. Use set_mask before calling sample()' + Console.ENDC
        else:
            self.set_mask(mask)

    def set_mask(self, mask):

        assert len(mask.shape) == 2, 'The mask must be 2D'

        self.mask = mask
        (rows, cols) = np.where(self.mask > 0)  # list of coordinates
        self.coords = zip(rows, cols)
        self.L = len(self.coords)
        print Console.OKGREEN + ' Mask set for the sampler' + Console.ENDC

    def check_convex(self, row, col):

        r1 = row - self.WSIDE
        r2 = row + self.WSIDE
        c1 = col - self.WSIDE
        c2 = col + self.WSIDE

        coords = self.coords
        return (r1, c1) in coords and (r2, c2) in coords and (r2, c1) in coords and (r1, c2) in coords

    def check_edge(self, row, col, xrows, xcols):

        r1 = row - xrows * self.WSIDE
        r2 = row + xrows * self.WSIDE
        c1 = col - xcols * self.WSIDE
        c2 = col + xcols * self.WSIDE

        coords = self.coords
        return (r1, c1) in coords or (r2, c2) in coords or (r2, c1) in coords or (r1, c2) in coords

    # def check_distance(self, row, col, xmax, ymax, step):
    #
    #     r1 = row - self.WSIDE
    #     r2 = row + self.WSIDE
    #     c1 = col - self.WSIDE
    #     c2 = col + self.WSIDE
    #     check = self._check_d
    #     return (check(r1, c1, xmax, ymax, step)
    #             or check(r1, c2, xmax, ymax, step)
    #             or check(r2, c1, xmax, ymax, step)
    #             or check(r2, c2, xmax, ymax, step))

    def check_distance(self, row, col, xmax, ymax, step):

        coords = self.coords
        for i in range(xmax):
            xDown = row + (i * step)
            xUp = row - (i * step)
            if (xDown, col) in coords or (xUp, col) in coords:
                if (row, col) not in coords:
                    return True

        for j in range(ymax):
            yRight = col + (j * step)
            yLeft = col - (j * step)
            if (row, yRight) in coords or (row, yLeft) in coords:
                if (row, col) not in coords:
                    return True

        return False

    def sample(self):
        """
        :param type: type of sampling
        :param callback: function to callback when a match is found (need to receive x,y coordinates)
        """
        type = self.type
        params = self.params
        callback = self.callback
        method = self.factory.get(type, self.overlap_sampling)

        print '\n\nInitiating sampling'
        print '-------------------------------'
        print 'Type               : ' + Console.BOLD + type + Console.ENDC
        print 'Window size        : %d' % self.WSIZE
        print 'Candidate list size: %d' % self.L

        return method(params, callback)

    def montecarlo_sampling(self, params, callback):

        if params == None:
            params = {}
        C = float(params.get('coverage', 0.5))
        edges_flag = params.get('edges', False)
        xrows = params.get('xrows', 0)
        xcols = params.get('xcols', 0)

        print 'Coverage           : %.1f' % (C * 100)
        N = int(self.L * C)
        print 'Number of samples  : ' + Console.OKBLUE + '%d' % N + Console.ENDC

        idx = random.sample(range(self.L), N)  # candidate indices
        selected = []

        print 'One moment please ....'
        for i in idx:
            (row, col) = self.coords[i]
            if self.check_edge(row, col, 0, 0):
                selected.append((row, col))

        print 'Number of selections: ' + Console.OKBLUE + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def overlap_sampling(self, params, callback):

        selected = []

        if params == None:
            params = {}

        edges_flag = params.get('edges', False)
        xrows = params.get('xrows', 1)
        xcols = params.get('xcols', 1)
        overlap_factor = int(params.get('overlap_factor', 2))

        print 'Overlap factor     : ' + Console.OKBLUE + '%d' % overlap_factor + Console.ENDC
        print 'Including edges?   : %s' % edges_flag

        if edges_flag:
            print '  etra rows        : %d' % xrows
            print '  extra columns    : %d' % xcols

        row, col = self.coords[0]  # numpy first dimension is rows
        height, width = self.mask.shape

        STEP = int(self.WSIZE / overlap_factor)
        print 'Step               : %d' % STEP

        print 'One moment please ....'

        while (row <= height and col <= width):
            while (col < width):

                if (row, col) in self.coords:
                    selected.append((row, col))
                elif edges_flag and self.check_edge(row, col, xrows, xcols):
                    selected.append((row, col))

                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + Console.BOLD + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def overlap_sampling(self, params, callback):

        selected = []

        if params == None:
            params = {}

        edges_flag = params.get('edges', False)
        xrows = params.get('xrows', 1)
        xcols = params.get('xcols', 1)
        overlap_factor = int(params.get('overlap_factor', 2))

        print 'Overlap factor     : ' + Console.OKBLUE + '%d' % overlap_factor + Console.ENDC
        print 'Including edges?   : %s' % edges_flag

        if edges_flag:
            print '  extra rows       : %d' % xrows
            print '  extra columns    : %d' % xcols

        row, col = self.coords[0]  # numpy first dimension is rows
        height, width = self.mask.shape

        STEP = int(self.WSIZE / overlap_factor)
        print 'Step               : %d' % STEP

        print 'One moment please ....'

        while (row <= height and col <= width):
            while (col < width):

                if (row, col) in self.coords:
                    selected.append((row, col))
                elif edges_flag and self.check_edge(row, col, xrows, xcols):
                    selected.append((row, col))

                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + Console.BOLD + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def background_sampling(self, params, callback):

        selected = []

        if params == None:
            params = {}

        xmax = params.get('xmax', 2)
        ymax = params.get('ymax', 2)
        overlap_factor = int(params.get('overlap_factor', 2))

        print 'Overlap factor     : ' + Console.OKBLUE + '%d' % overlap_factor + Console.ENDC
        print 'xmax               : %d' % xmax
        print 'ymax               : %d' % ymax

        height, width = self.mask.shape
        STEP = int(self.WSIZE / overlap_factor)
        print 'Step               : %d' % STEP
        print 'One moment please ....'

        row, col = 0, 0
        while (row <= height and col <= width):
            while (col < width):
                if self.check_distance(row, col, xmax, ymax, STEP):
                    selected.append((row, col))
                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + Console.BOLD + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def convex_sampling(self, params, callback):

        selected = []

        if params == None:
            params = {}

        overlap_factor = params.get('overlap_factor', 2)

        print 'Overlap factor     : ' + Console.OKBLUE + '%.2f' % overlap_factor + Console.ENDC

        row, col = self.coords[0]  # numpy first dimension is rows
        height, width = self.mask.shape

        STEP = self.WSIZE / overlap_factor

        print 'One moment please ....'

        while (row <= height and col <= width):
            while (col < width):

                if (row, col) in self.coords:
                    if self.check_convex(row, col):
                        selected.append((row, col))
                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + Console.BOLD + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected

    def all_sampling(self, params, callback):

        selected = []

        row, col = 0, 0  # self.coords[0]  # numpy first dimension is rows
        height, width = self.mask.shape

        STEP = self.WSIZE / 2

        print 'One moment please ....'

        while (row <= height and col <= width):
            while (col < width):
                selected.append((row, col))
                col = col + STEP
            row = row + STEP
            col = 0

        print 'Number of selections: ' + Console.OKBLUE + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (row, col) = s
            if callback is not None: callback(row, col)

        return selected
