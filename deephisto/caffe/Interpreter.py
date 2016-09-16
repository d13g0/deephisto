import cmd
import matplotlib.pylab as plt
import multiprocessing as mlt
from NetTest import NetTest

class Interpreter(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.obs = NetTest()

    prompt = '>> '
    ruler = '-'

    def emptyline(self):
        pass

    def do_exit(self, s):
        return True

    def help_exit(self):
        print "Exit the interpreter."
        print "You can also use the Ctrl-D shortcut."

    def do_intro(self, line):
        print 'Welcome to the network observer. Type intro to show this message'
        print '------------------------------------------------------------------------------'
        print
        print ' Commands: '
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
        print '   anim [directory] [start] [stop]  [step]  : sets up an animation of the net through its epochs'
        print '   next                        : next epoch   '
        print '   play (loop)                 : plays animation'
        print
        print '   exit: exits the network observer'
        print '------------------------------------------------------------------------------'

    def do_load(self, args):
        """load [directory] [epoch] : loads a network with the given epoch data"""
        if (len(args.split()) != 2):
            print 'Wrong number of paramters. [directory] [epoch] expected.'
            return

        directory, epoch = args.split()
        epoch = int(epoch)
        self.obs.load_model(directory, epoch)

    def do_ping(self, args):
        """ping [patch (optional)] if patch is present queries the network for this patch. otherwise queries a random patch"""
        if len(args.split()) == 0:
            patch = None
        else:
            patch = args
        image_file, label, pred = self.obs.get_single_prediction(patch_name=patch)
        process = mlt.Process(target=self.obs.show_single_prediction, args=(image_file, label, pred))
        process.start()
        # self.obs.single_prediction(patch_name=patch)

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
            self.obs.load_model(directory, epoch)

        self.obs.setup_panel()
        self.do_rand(None)
        plt.tight_layout()

    def do_rand(self, args):
        """rand : loads a random sample of patches, queries the network and displays the ground truth (GT) and the prediction
                PR in the panel"""
        obs = self.obs
        obs.get_inputs()
        obs.get_predictions()
        obs.show_labels()
        obs.show_predictions()

    def do_epoch(self, args):
        """epoch [e] : loads the epoch indicated by e and updates the pannel"""
        if len(args) != 1:
            print "Please indicate the epoch you want to see [integer]"
            return

        epoch = int(args)
        obs = self.obs
        obs.load_model(obs.directory, epoch)
        obs.get_predictions()
        obs.show_predictions()

    def do_anim(self, args):
        """sets up an animation of the net through its epoch
        :param directory: the data directory
        :param start: initial epoch
        :param stop:  final epoch
        :param step: animation step
        """
        if len(args.split()) != 4:
            print 'Wrong number of parameters please check'
            return
        directory, start, stop, step = args.split()
        start = int(start)
        stop = int(stop)
        step = int(step)
        obs = self.obs
        obs.configure(directory, start, stop, step)
        obs.setup_panel()
        obs.get_predictions()
        obs.show_predictions()
        plt.tight_layout()

    def do_next(self, args):
        """next: Shows the next epoch in the animation"""
        obs = self.obs
        obs.next_epoch()
        obs.get_predictions()
        obs.show_predictions()
        plt.pause(0.000001)

    def do_play(self, args):
        """play: Shows the animation. If the parameter loop is present, it loops back to the start when it finishes."""
        if len(args) == 0:
            loop = False
        elif len(args) == 1 and args.trim().lower() == 'loop':
            loop = True
        obs = self.obs
        obs.show_predictions()
        plt.tight_layout()

        while obs.next_epoch(loop):
            obs.get_predictions()
            obs.show_predictions()
            plt.pause(0.00001)
        print 'DONE'

    def do_pipe(self, args):
        for arg in args:
            s = arg
            self.onecmd(s)

    def parseline(self, line):
        if '|' in line:
            return 'pipe', line.split('|'), line
        return cmd.Cmd.parseline(self, line)


if __name__ == '__main__':
    i = Interpreter()
    i.do_intro(None)
    i.cmdloop()
