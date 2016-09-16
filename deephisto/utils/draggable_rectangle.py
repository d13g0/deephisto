# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:40:25 2016

@author: dcantor
"""

from deephisto.patch.patch_sampler import PatchSampler as Sampler

class DraggableRectangle:
    def __init__(self, rect, callback):
        self.rect = rect
        self.press = None
        self.callback = callback
        self.centre_x, self.centre_y = self.rect.get_xy()
        self.centre_x  += Sampler.WSIDE
        self.centre_y  += Sampler.WSIDE

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.centre_x = x0 + Sampler.WSIDE
        self.centre_y = y0 + Sampler.WSIDE
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        self.centre_x = x0+dx + Sampler.WSIDE
        self.centre_y = y0+dy + Sampler.WSIDE
        self.callback(int(self.centre_x), int(self.centre_y)) #callback must be the center coords

        #self.rect.figure.canvas.draw()
        


    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.rect.figure.canvas.draw()
        

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)