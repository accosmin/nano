import os

class config:
        def __init__(self):
                # useful paths
                homedir = os.path.expanduser('~')
                self.expdir = homedir + "/experiments/results"
                self.dbdir = homedir + "/experiments/databases"

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_info = crtpath + "/../build-release/apps/info"
                self.app_train = crtpath + "/../build-release/apps/train"
                self.app_stats = crtpath + "/../build-release/apps/stats"
                self.app_tabulate = crtpath + "/../build-release/apps/tabulate"

        # available losses: {name, command line parameters}+
        def losses(self):
                return {
                        "cauchy" : "--loss cauchy",
                        "classnll" : "--loss classnll",
                        "exponential" : "--loss exponential",
                        "logistic" : "--loss logistic",
                        "square" : "--loss square"
                }

        # available iterators: {name, command line parameters}+
        def iterators(self):
                return {
                        "default" : "--iterator default",
                        "noise" : "--iterator noise",
                        "warp" : "--iterator warp"
                }

        # training methods: {name, command line parameters}+
        def trainers(self):
                return {
                        "stoch_ag" : "--trainer stoch --trainer-params opt=ag",
                        "stoch_agfr" : "--trainer stoch --trainer-params opt=agfr",
                        "stoch_aggr" : "--trainer stoch --trainer-params opt=aggr",
                        "stoch_sg" : "--trainer stoch --trainer-params opt=sg",
                        "stoch_sgm" : "--trainer stoch --trainer-params opt=sgm",
                        "stoch_ngd" : "--trainer stoch --trainer-params opt=ngd",
                        "stoch_svrg" : "--trainer stoch --trainer-params opt=svrg",
                        "stoch_asgd" : "--trainer stoch --trainer-params opt=asgd",
                        "stoch_adagrad" : "--trainer stoch --trainer-params opt=adagrad",
                        "stoch_adadelta" : "--trainer stoch --trainer-params opt=adadelta",
                        "stoch_adam" : "--trainer stoch --trainer-params opt=adam",
                        "stoch_rmsprop" : "--trainer stoch --trainer-params opt=rmsprop",

                        "batch_lbfgs" : "--trainer batch --trainer-params opt=lbfgs",
                        "batch_cgd" : "--trainer batch --trainer-params opt=cgd",
                        "batch_gd" : "--trainer batch --trainer-params opt=gd"
                }

        # task parameters
        def task_synth(self, name, params):
                return "--task synth-{0} --task-params {1}".format(name, params)

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return self.task_synth("charset", "type={0},color={1},irows={2},icols={3},count={4}".format(ctype, color, irows, icols, count))

        def task(self, name):
                return "--task {0} --task-params dir={1}".format(name, self.dbdir + "/" + name)

        def task_mnist(self):
                return self.task("mnist")

        def task_cifar10(self):
                return self.task("cifar10")

        def task_iris(self):
                return self.task("iris")

        def task_wine(self):
                return self.task("wine")
