import os

class config:
        def __init__(self):
                # useful paths
                homedir = os.path.expanduser('~')
                self.expdir = homedir + "/experiments/results"
                self.dbdir = homedir + "/experiments/databases"

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_train = crtpath + "/../build-release/apps/train"
                self.app_stats = crtpath + "/../build-release/apps/stats"
                self.app_info = crtpath + "/../build-release/apps/info"

        # available losses: {name, command line parameters}+
        def losses(self):
                return {
                        "loss_cauchy" : "--loss cauchy",
                        "loss_classnll" : "--loss classnll",
                        "loss_exponential" : "--loss exponential",
                        "loss_logistic" : "--loss logistic",
                        "loss_square" : "--loss square"
                }

        # available criteria: {name, command line parameters}+
        def criteria(self):
                return {
                        "crit_avg" : "--criterion avg",
                        "crit_avg_l2n" : "--criterion avg-l2n",
                        "crit_avg_var" : "--criterion avg-var",
                        "crit_max" : "--criterion max",
                        "crit_max_l2n" : "--criterion max-l2n",
                        "crit_max_var" : "--criterion max-var"
                }

        # training policies: {name, command line parameters}+
        def policies(self):
                return {
                        "stop_early" : ",policy=stop_early",
                        "all_epochs" : ",policy=all_epochs"
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

                        "batch_lbfgs" : "--trainer batch --trainer-params opt=lbfgs",
                        "batch_cgd" : "--trainer batch --trainer-params opt=cgd",
                        "batch_gd" : "--trainer batch --trainer-params opt=gd"
                }

        # task parameters
        def task_synth(self, name, params):
                return "--task synth-{0} --task-params {1}".format(name, params)

        def task(self, name):
                return "--task {0} --task-params dir={1}".format(name, self.dbdir + "/" + name)

        def task_synth_sign(self, isize = 100, osize = 10, count = 10000, noise = 1e-4):
                return self.task_synth("sign", "isize={0},osize={1},count={2},noise={3}".format(isize, osize, count, noise))

        def task_synth_affine(self, isize = 100, osize = 10, count = 10000, noise = 1e-4):
                return self.task_synth("affine", "isize={0},osize={1},count={2},noise={3}".format(isize, osize, count, noise))

        def task_synth_matmul(self, irows = 23, icols = 27, count = 10000, noise = 1e-4):
                return self.task_synth("matmul", "irows={0},icols={1},count={2},noise={3}".format(irows, icols, count, noise))

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return self.task_synth("charset", "type={0},color={1},irows={2},icols={3},count={4}".format(ctype, color, irows, icols, count))

        def task_mnist(self):
                return self.task("mnist")
