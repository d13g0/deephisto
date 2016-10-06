import os

os.environ['GLOG_minloglevel'] = '2'
import cmd
import matplotlib.pylab as plt
import multiprocessing as mlt

from deephisto import Locations, Console
from deephisto.caffe import NetInteractor, NetTest


class Interpreter(cmd.Cmd):
    def __init__(self, locations):
        cmd.Cmd.__init__(self)
        self.locations = locations
        self.obs = NetInteractor()
        self.net_test = NetTest(locations)
        self.data_dir = None

    prompt = '>> '
    ruler = '-'

    def emptyline(self):
        pass

    def do_exit(self, s):
        return True

    def help_exit(self):
        print "Exit the interpreter."
        print "You can also use the Ctrl-D shortcut."

    def do_man(self, line):
        print 'Welcome to the network observer. Type ' + Console.BOLD + 'man' + Console.ENDC + ' to show this message'
        print '------------------------------------------------------------------------------'
        print
        print ' Commands: '
        print
        print '   data  [dir]                 : sets the data directory '
        print
        print '   Single patch:'
        print '   ============='
        print '   load [directory]  [epoch]   : loads a trained network'
        print '   ping [patch]                : gets a random prediction from the validation set'
        print '   net                         : shows the structure of the network'
        print
        print
        print '   Panel:'
        print '   ======'
        print '   peek [directory] [epoch]    : loads the panel for a given epoch'
        print '   epoch [e]                   : sets the current epoch to e'
        print '   rand                        : changes the data in the panel'
        print
        print
        print '   Animation:'
        print '   =========='
        print '   next                                     : next epoch   '
        print '   play [directory] [start] [stop]  [step]  : sets up an animation of the net through its epochs'
        print
        print
        print '   Subject:'
        print '   ========'
        print '   run [subject] [slice] [avg|max|med|mod] : runs the network on this slice patcy-by-patch'
        print
        print '   man                                     : shows this message                          '
        print '   exit                                    : exits the interpreter'
        print '------------------------------------------------------------------------------'

    def do_data(self, args):
        """Sets the data source for the network"""
        if (len(args.split()) != 1):
            print 'Wrong number of paramters. data [dir] expected.'
            return
        self.data_dir = args.split()[0]

        dir = self.locations.PATCHES_DIR + '/' + self.data_dir
        if not os.path.exists(dir):
            print Console.WARNING + '%s does not exist'%self.data_dir + Console.ENDC
            return
        dir = self.locations.PATCHES_DIR + '/' + self.data_dir

        if not os.path.exists(dir):
            print '%s does not exist'
            return

        print 'data dir set to ' + Console.BOLD +'%s' % self.data_dir + Console.ENDC


        train, test = self.obs.load_lists(data_dir=self.data_dir)
        print
        print 'size training set    :   %d'%len(train)
        print 'size validation set  :   %d'%len(test)
        print

    def do_load(self, args):
        """load [directory] [epoch] : loads a network with the given epoch data"""
        if (len(args.split()) != 2):
            print ' Wrong number of paramters. [directory] [epoch] expected.'
            return

        directory, epoch = args.split()
        epoch = int(epoch)

        if self.data_dir == None:
            print ' you need to set the data directory first with set_data [dir]'
            return
        else:
            self.obs.load_model(directory, epoch, self.data_dir)

    def do_ping(self, args):
        """ping [patch (optional)] if patch is present queries the network for this patch. otherwise queries a random patch"""
        if len(args.split()) == 0:
            patch = None
        else:
            patch = args
        image_file, label, pred, channels = self.obs.get_single_prediction(patch_name=patch)
        plt.ion()
        self.obs.show_single_prediction(image_file, label, pred, channels)

    def do_net(self, args):
        """net: shows the structure of the current network"""
        self.obs.show_network_model();

    def do_peek(self, args):
        """peek [directory] [epoch] :  oads a sample panel for the given directory and epoch"""
        if (len(args.split()) != 2):
            if self.obs.directory is None and self.obs.epoch is None:
                print 'Wrong number of paramters. [directory] [epoch] expected.'
                return
        else:
            directory, epoch = args.split()
            epoch = int(epoch)
            if self.data_dir == None:
                print 'You need to set the data directory first with set_data [dir]'
                return
            else:
                try:
                    self.obs.load_model(directory, epoch, self.data_dir)
                except Exception as e:
                    print e.message
                    return


        self.obs.setup_panel()
        self.do_rand(None)

    def do_rand(self, args):
        """rand : loads a random sample of patches, queries the network and displays the ground truth (GT) and the prediction
                PR in the panel"""
        obs = self.obs
        plt.ion()
        obs.get_inputs()
        obs.get_predictions()
        obs.show_labels()
        plt.tight_layout()
        obs.show_predictions()
        plt.draw()

    def do_epoch(self, args):
        """epoch [e] : loads the epoch indicated by e and updates the pannel"""
        if len(args.split()) != 1:
            print "Please indicate the epoch you want to see [integer]"
            return

        epoch = int(args)
        obs = self.obs
        try:
            obs.load_model(obs.directory, epoch, self.data_dir)
        except:
            print 'Error'
            return

        obs.get_predictions()
        obs.show_predictions()


    def do_next(self, args):
        """next: Shows the next epoch in the animation"""
        obs = self.obs
        obs.next_epoch()
        obs.get_predictions()
        obs.show_predictions()
        plt.pause(0.000001)

    def do_play(self, args):
        """sets up an animation of the net through its epoch
                :param directory: the data directory
                :param start: initial epoch
                :param stop:  final epoch
                :param step: animation step
        """
        import matplotlib.animation as anim

        if len(args.split()) != 4:
            print 'Wrong number of parameters please check'
            return

        if self.data_dir == None:
            print 'You need to set the data directory first with set_data [dir]'
            return


        obs = self.obs
        directory, start, stop, step = args.split()
        start = int(start)
        stop = int(stop)
        step = int(step)
        if step < 0:
            print '[step] cannot be a negative number'
            return

        if stop < start:
            print '[stop] needs to be higher than [start]'
            return


        folder = self.locations.MOVIE_DIR + ('/%s_%d_%d_%d_%s'%(directory, start, stop, step, self.data_dir))
        self.locations.check_dir_of(folder + '/dummy')

        obs = self.obs
        obs.verbose = False
        obs.set_animation_params(directory, start, stop, step, self.data_dir)

        plt.ion()
        self.obs.setup_panel()
        obs.get_inputs()
        obs.show_labels()
        plt.tight_layout()
        flag_loop = True

        steps = int((stop - start)/step)

        L = steps-1

        for i in range(steps):
            print 'Frame %d from %d'%(i, L)
            try:
                obs.get_predictions()
                obs.show_predictions()
                plt.pause(0.000001)
                plt.savefig(folder+'/frame%03d'%i, frameon=None, facecolor=obs.fig.get_facecolor(), edgecolor='none')
                obs.next_epoch(False)
            except KeyboardInterrupt:
                pass

        obs.verbose = True
        print 'DONE'

    def do_run(self, args):
        if (len(args.split()) != 3):
            print 'Error: run [subject] [slice] [avg|max|med|mod]'
            return
        subject, slice, blend = args.split()
        slice = int(slice)

        try:
            self.net_test.load_data(subject, slice)
            self.net_test.load_network(self.obs.directory, self.obs.epoch, self.data_dir)
            self.net_test.set_window(28)  #hard code. Patches are 28x28
            self.net_test.set_blend(blend)
            self.net_test.go()
        except (ValueError, AssertionError) as e:
            print e
            return


    def do_pipe(self, args):
        for arg in args:
            s = arg
            self.onecmd(s)

    def parseline(self, line):
        if '|' in line:
            return 'pipe', line.split('|'), line
        return cmd.Cmd.parseline(self, line)


if __name__ == '__main__':
    locations = Locations('/home/dcantor/projects/deephisto')
    i = Interpreter(locations)
    i.do_man(None)
    i.cmdloop()
