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
                        "sexponential" : "--loss s-exponential",
                        "mexponential" : "--loss m-exponential",
                        "slogistic" : "--loss s-logistic",
                        "mlogistic" : "--loss m-logistic",
                        "square" : "--loss square"
                }

        # configure iterator with the given name and parameters
        def config_loss(self, name, parameters = ""):
                return self.losses().get(name)

        # available iterators: {name, command line parameters}+
        def iterators(self):
                return {
                        "default" : "--iterator default",
                        "noise" : "--iterator noise",
                        "warp" : "--iterator warp"
                }

        # configure iterator with the given name and parameters
        def config_iterator(self, name, parameters = "none"):
                return self.iterators().get(name) + " --iterator-params " + parameters

        # training methods: {name, command line parameters}+
        def trainers(self):
                return {
                        "stoch_ag" : "--trainer stoch --trainer-params solver=ag",
                        "stoch_agfr" : "--trainer stoch --trainer-params solver=agfr",
                        "stoch_aggr" : "--trainer stoch --trainer-params solver=aggr",
                        "stoch_sg" : "--trainer stoch --trainer-params solver=sg",
                        "stoch_sgm" : "--trainer stoch --trainer-params solver=sgm",
                        "stoch_ngd" : "--trainer stoch --trainer-params solver=ngd",
                        "stoch_svrg" : "--trainer stoch --trainer-params solver=svrg",
                        "stoch_asgd" : "--trainer stoch --trainer-params solver=asgd",
                        "stoch_adagrad" : "--trainer stoch --trainer-params solver=adagrad",
                        "stoch_adadelta" : "--trainer stoch --trainer-params solver=adadelta",
                        "stoch_adam" : "--trainer stoch --trainer-params solver=adam",
                        "stoch_rmsprop" : "--trainer stoch --trainer-params solver=rmsprop",

                        "batch_lbfgs" : "--trainer batch --trainer-params solver=lbfgs",
                        "batch_cgd" : "--trainer batch --trainer-params solver=cgd",
                        "batch_gd" : "--trainer batch --trainer-params solver=gd"
                }

        # configure trainer with the given name and parameters
        def config_trainer(self, name, parameters = ""):
                return self.trainers().get(name) + "," + parameters

        # configure synthetic tasks
        def task_synth(self, name, params):
                return "--task synth-{0} --task-params {1}".format(name, params)

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return self.task_synth("charset", "type={0},color={1},irows={2},icols={3},count={4}".format(ctype, color, irows, icols, count))

        # configure tasks
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
