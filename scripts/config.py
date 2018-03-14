import os

class config:
        """ utility to manage the configuration needed for an experiment:
                - paths (e.g. datasets, results, utilities)
                - available ML objects (e.g. solvers, loss functions)
        """
        def __init__(self, build_folder = "../build-release"):
                homedir = os.path.expanduser('~')
                self.expdir = os.path.join(homedir, "experiments", "results")
                self.dbdir = os.path.join(homedir, "experiments", "databases")

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_info = os.path.join(crtpath, build_folder, "apps", "info")
                self.app_train = os.path.join(crtpath, build_folder, "apps", "train")
                self.app_stats = os.path.join(crtpath, build_folder, "apps", "stats")
                self.app_tabulate = os.path.join(crtpath, build_folder, "apps", "tabulate")

        def losses(self):
                """ available loss functions"""
                return ["cauchy",               # regression (robust to noise)
                        "square",               # regression
                        "classnll",             # classification (single label)
                        "s-logistic",           # classification (single label)
                        "m-logistic",           # classification (multi label)
                        "s-exponential",        # classification (single label)
                        "m-exponential"]        # classification (multi label)

        def activations(self):
                """ available activation functions """
                return ["act-unit",
                        "act-sin",      # [-1, +1]
                        "act-tanh",     # [-1, +1]
                        "act-splus",    # [ 0,  1]
                        "act-snorm",    # [-1, +1]
                        "act-ssign",    # [-1, +1]
                        "act-sigm",     # [ 0,  1]
                        "act-pwave"]    # [-1, +1]

        def stoch_solvers(self):
                """ available stochastic solvers """
                return ["ag", "agfr", "aggr",
                        "sg", "sgm", "ngd", "svrg", "asgd", "cocob",
                        "adagrad", "adadelta", "adam", "rmsprop", "amsgrad"]

        def batch_solvers(self):
                """ available batch (line-search) solvers """
                return ["gd", "cgd", "lbfgs"]

        def stoch_trainer(self, solver, epochs = 100, patience = 10, epsilon = 1e-6):
                """ create a stochastic trainer """
                assert(solver in self.stoch_solvers())
                return {"type": "stoch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon}

        def batch_trainer(self, solver, epochs = 100, patience = 10, epsilon = 1e-6):
                """ create a batch trainer """
                assert(solver in self.batch_solvers())
                return {"type": "batch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon}

        def loss(self, loss):
                """ create a loss """
                assert(loss in self.losses())
                return {"type": loss}

        def task(self, name):
                """ create a task """
                return {"type": name, "dir": os.path.join(self.dbdir, name)}

        def task_mnist(self):
                return self.task("mnist")

        def task_cifar10(self):
                return self.task("cifar10")

        def task_iris(self):
                return self.task("iris")

        def task_wine(self):
                return self.task("wine")

        def task_synth_nparity(self, n = 32, count = 10000):
                return {"type": "synth-nparity", "n": n, "count": count}

        def task_synth_affine(self, isize = 32, osize = 32, noise = 0.0, count = 10000):
                return {"type": "synth-affine", "isize": isize, "osize": osize, "noise": noise, "count": count}

        def task_synth_peak2d(self, irows = 32, icols = 32, noise = 0.0, count = 10000):
                return {"type": "synth-peak2d", "irows": irows, "icols": icols, "noise": noise, "count": count}
