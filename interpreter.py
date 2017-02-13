import os, pdb

os.environ['GLOG_minloglevel'] = '2'
import cmd
import matplotlib.pylab as plt
import multiprocessing as mlt

from steps.config import dh_config_selector
from deephisto import Locations, Console
from deephisto.net import NetInteractor, NetTest


class Interpreter(cmd.Cmd):

    def __init__(self, config):
        cmd.Cmd.__init__(self)
        self.config = config
        #self.locations = Locations(config)
        self.net_interactor = NetInteractor(config)
        self.net_test = NetTest(config)
        self.patch_dir = None
        self.dataset_dir = None

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
        print 'Welcome to Deep Histo. Type ' + Console.BOLD + 'man' + Console.ENDC + ' to show this message'
        print '------------------------------------------------------------------------------'
        print
        print ' Commands: '
        print
        print Console.BOLD + '   Dataset' + Console.ENDC
        print
        print '   - dataset [dir]               : provides information about the current dataset. Sets the new dataset if the paramter [dir] is present'
        print
        print Console.BOLD + '   Single patch' + Console.ENDC
        print
        print '   - load [directory]  [epoch]   : loads a trained network'
        print '   - ping [patch]                : gets a random prediction from the validation set'
        print '   - net                         : shows the structure of the network'
        print
        print Console.BOLD + '   Panel' + Console.ENDC
        print
        print '   - peek [directory] [epoch]    : loads the panel for a given epoch'
        print '   - epoch [e]                   : sets the current epoch to e'
        print '   - rand                        : changes the data in the panel'
        print
        print Console.BOLD + '   Animation' + Console.ENDC
        print
        print '   - next                                     : next epoch   '
        print '   - play [directory] [start] [stop]  [step]  : sets up an animation of the net through its epochs'
        print
        print Console.BOLD + '   Subject' + Console.ENDC
        print
        print '   - run [subject] [slice] [avg|max|med|mod] : runs the network on this slice patcy-by-patch'
        print
        print '   - man                                     : shows this message                          '
        print '   - exit                                    : exits the interpreter'
        print '------------------------------------------------------------------------------'

    def load_defaults_from_config(self):
        self.patch_dir = self.config.PATCH_DIR
        self.dataset_dir = self.config.DATASET_DIR
        print
        print 'Loading defaults from configuration file:'
        self.show_dataset()
        print

    def show_dataset(self):

        print 'Patch   dir : ' + Console.BOLD + '%s' % self.patch_dir + Console.ENDC
        print 'Dataset dir :'  + Console.BOLD + '%s' % self.dataset_dir + Console.ENDC


        train, test = self.net_interactor.load_dataset_metadata(dataset_dir=self.dataset_dir)
        print
        print 'size training set    :   %d' % len(train)
        print 'size validation set  :   %d' % len(test)
        print
        self.show_dataset_split(train,test)

    def show_dataset_split(self, train, test):
        """
        WARNING
        This method is hardcoded to the patch template under the
        [patches] section in the configuration file. Changes
        to the template will BREAK this method
        """
        train_set = set()
        test_set = set()

        for sample in train:
            _,_,pid,slice_id,_,_ = sample.split('_')
            slice = 'EPI_'+pid+'_'+slice_id
            train_set.add(slice)

        for sample in test:
            _,_,pid,slice_id,_,_ = sample.split('_')
            slice = 'EPI_'+pid+'_'+slice_id
            test_set.add(slice)

        train_set = sorted(train_set)
        test_set = sorted(test_set)

        print 'Slices in '+ Console.BOLD+'TRAINING'+ Console.ENDC+' set'
        for item in train_set:
            print item,

        print
        print
        print 'Slices in '+ Console.BOLD+'TESTING'+ Console.ENDC+' set'
        for item in test_set:
            print item,
        print
        print

    def do_dataset(self,args):
        """
        Displays information about the current dataset
        """
        if len(args.split()) == 0:
            self.show_dataset()
            return

        dir = args.split()[0]
        patch_dir = os.path.join(os.path.dirname(self.config.PATCH_DIR), dir)
        dataset_dir = os.path.join(os.path.dirname(self.config.DATASET_DIR), dir)

        if not os.path.exists(patch_dir):
            print Console.WARNING + '%s does not exist'%patch_dir + Console.ENDC
            return

        if not os.path.exists(dataset_dir):
            print Console.WARNING + '%s does not exist' % datset_dir + Console.ENDC
            return

        self.patch_dir = patch_dir
        self.dataset_dir = dataset_dir
        self.show_dataset()








    def do_load(self, args):
        """
        load [directory] [epoch] : loads a network with the given epoch data
        """
        if (len(args.split()) != 2):
            print ' Wrong number of paramters. [directory] [epoch] expected.'
            return

        directory, epoch = args.split()
        epoch = int(epoch)
        self.net_interactor.load_model(directory, epoch, self.patch_dir, self.dataset_dir)

    def do_ping(self, args):
        """ping [patch (optional)] if patch is present queries the network for this patch. otherwise queries a random patch"""
        if len(args.split()) == 0:
            patch = None
        else:
            patch = args
        image_file, label, pred, channels = self.net_interactor.get_single_prediction(patch_name=patch)
        plt.ion()
        self.net_interactor.show_single_prediction(image_file, label, pred, channels)

    def do_net(self, args):
        """net: shows the structure of the current network"""
        self.net_interactor.show_network_model();

    def do_peek(self, args):
        """peek [directory] [epoch] :  oads a sample panel for the given directory and epoch"""
        if (len(args.split()) != 2):
            if self.net_interactor.directory is None and self.net_interactor.state is None:
                print 'Wrong number of paramters. [directory] [epoch] expected.'
                return
        else:
            directory, epoch = args.split()
            epoch = int(epoch)
            if self.patch_dir == None:
                print 'You need to set the data directory first with set_data [dir]'
                return
            else:
                try:
                    self.net_interactor.load_model(directory, epoch, self.patch_dir)
                except Exception as e:
                    print e.message
                    return


        self.net_interactor.setup_panel()
        self.do_rand(None)

    def do_rand(self, args):
        """rand : loads a random sample of patches, queries the network and displays the ground truth (GT) and the prediction
                PR in the panel"""
        obs = self.net_interactor
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
        obs = self.net_interactor
        try:
            obs.load_model(obs.directory, epoch, self.patch_dir)
        except:
            print 'Error'
            return

        obs.get_predictions()
        obs.show_predictions()


    def do_next(self, args):
        """next: Shows the next epoch in the animation"""
        obs = self.net_interactor
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

        if self.patch_dir == None:
            print 'You need to set the data directory first with set_data [dir]'
            return


        obs = self.net_interactor
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


        folder = self.locations.MOVIE_DIR + ('/%s_%d_%d_%d_%s' % (directory, start, stop, step, self.patch_dir))
        self.locations.check_dir_of(folder + '/dummy')

        obs = self.net_interactor
        obs.verbose = False
        obs.set_animation_params(directory, start, stop, step, self.patch_dir)

        plt.ion()
        self.net_interactor.setup_panel()
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
            self.net_test.load_network(self.net_interactor.directory,
                                       self.net_interactor.state,
                                       self.dataset_dir)
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
    cfile = dh_config_selector()
    i = Interpreter(cfile)
    i.load_defaults_from_config()
    i.do_man(None)
    i.cmdloop()
