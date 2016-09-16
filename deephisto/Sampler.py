import numpy as np
import random

from .Console import Console

class Sampler:

    WSIZE = 30
    WSIDE = WSIZE / 2
    S_MONTECARLO = 'Montecarlo'
    S_SYSTEMATIC = 'Systematic'
    S_OVERLAP = 'Overlap'


    def __init__(self, bmask):

        self.set_mask(bmask)

        self.samplers = {
            Sampler.S_MONTECARLO: self.montecarlo_sampling,
            Sampler.S_SYSTEMATIC: self.systematic_sampling,
            Sampler.S_OVERLAP   : self.regular_overlap_sampling,
        }


    def set_mask(self,bmask):
        self.bmask = bmask
        (a, b) = np.where(self.bmask > 0)  # list of coordinates
        self.coords = zip(a, b)
        self.L = len(self.coords)

    def check_convex(self, x,y):

        x1 = x - Sampler.WSIDE
        x2 = x + Sampler.WSIDE
        y1 = y - Sampler.WSIDE
        y2 = y + Sampler.WSIDE

        coords = self.coords
        return (x1, y1) in coords and (x2, y2) in coords


    def check_fully_convex(self, x,y):

        x1 = x - Sampler.WSIDE
        x2 = x + Sampler.WSIDE
        y1 = y - Sampler.WSIDE
        y2 = y + Sampler.WSIDE

        coords = self.coords

        rx = range(x1,x2+1)
        ry = range(y1,y2+1)
        for i,j in itertools.product(rx,ry):
            if (i,j) not in coords:
                return False

        return True


    def sample(self, type, params=None, callback=None):
        """
        :param type: type of sampling
        :param callback: function to callback when a match is found (need to receive x,y coordinates)
        """
        sampler = self.samplers.get(type, self.montecarlo_sampling)


        print '\n\nInitiating sampling'
        print '-------------------------------'
        print 'Type               : ' + Console.BOLD + type + Console.ENDC
        print 'Candidate list size: %d' % self.L

        return sampler(params,callback)


    def montecarlo_sampling(self, params,callback):

        if params is not None and ('C' in params):
            C = params['C']
        else:
            C = 50

        print 'Coverage           : %d%%'%C
        N = int(self.L*C/100)
        print 'Number of samples  : ' + Console.OKBLUE + '%d'%N + Console.ENDC


        idx = random.sample(range(self.L), N)  # candidate indices
        selected = []

        print 'One moment please ....'
        for i in idx:
            (x,y) = self.coords[i]
            if self.check_convex(x,y):
                selected.append((x,y))


        print 'Number of selections: ' + Console.OKBLUE + '%d'%len(selected) + Console.ENDC

        for s in selected:
            (x,y) = s
            if callback is not None: callback(x,y)

        return selected





    def systematic_sampling(self, params, callback):
        """
        Finds windows of a given size that fit in the mask, and returns the center pixel for each
        one of those windows.
        """
        selected = []
        for (x, y) in self.coords:
            if self.check_convex(x,y):
                selected.append((x,y))
                if callback is not None: callback(x,y)

        return selected

    def regular_overlap_sampling(self, params, callback):

        selected = []

        fx, fy = self.coords[0]
        selected.append((fx, fy))

        MAX_X, MAX_Y = self.bmask.shape

        STEP = Sampler.WSIZE / 5

        print 'One moment please ....'

        while(fx <= MAX_X and fy <=MAX_Y):
            while (fy < MAX_Y):
                fy = fy + STEP
                if (fx,fy) in self.coords:
                    selected.append((fx,fy))
                    # if callback is not None: callback(fx, fy)
            fx = fx+ STEP
            fy = 0



        print 'Number of selections: ' + Console.OKBLUE + '%d' % len(selected) + Console.ENDC

        for s in selected:
            (x,y) = s
            if callback is not None: callback(x,y)


        return selected