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
                self.app_builder = os.path.join(crtpath, build_folder, "apps", "builder")
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

        def model(self, model_type, conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type):
                """ create a model using the built-in command line utility apps/builder """
                return  "--{} --act-type {} --conv3d-param {} --affine-param {} "\
                        "--imaps {} --irows {} --icols {} --omaps {} --orows {} --ocols {} ".format(
                        model_type, act_type, ','.join(map(str, conv3d_param)), ','.join(map(str, affine_param)),
                        imaps, irows, icols, omaps, orows, ocols)

        def linear(self, imaps, irows, icols, omaps, orows, ocols):
                """ create a linear model """
                return  "--linear "\
                        "--imaps {} --irows {} --icols {} --omaps {} --orows {} --ocols {} ".format(
                        imaps, irows, icols, omaps, orows, ocols)

        def mlp(self, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type="act-snorm"):
                """ create a MLP (multi-layer perceptron) """
                return self.model("mlp", [], affine_param, imaps, irows, icols, omaps, orows, ocols, act_type)

        def cnn(self, conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type="act-snorm"):
                """ create a CNN (convolution neural network) """
                return self.model("cnn", conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type)

        def stoch_solvers(self):
                """ available stochastic solvers """
                return ["ag", "agfr", "aggr",
                        "sg", "sgm", "ngd", "svrg", "asgd",
                        "adagrad", "adadelta", "adam", "rmsprop"]

        def batch_solvers(self):
                """ available batch (line-search) solvers """
                return ["gd", "cgd", "lbfgs"]

        def stoch_trainer(self, solver, epochs = 100, patience = 10, epsilon = 1e-6):
                """ create a stochastic trainer """
                assert(solver in self.stoch_solvers())
                return {"trainer": "stoch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon}

        def batch_trainer(self, solver, epochs = 100, patience = 10, epsilon = 1e-6):
                """ create a batch trainer """
                assert(solver in self.batch_solvers())
                return {"trainer": "batch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon}

        def loss(self, loss):
                """ create a loss """
                assert(loss in self.losses())
                return {"loss": loss}

        def task(self, name):
                """ create a task """
                return {"task": name, "dir": os.path.join(self.dbdir, name)}

        def task_mnist(self):
                return self.task("mnist")

        def task_cifar10(self):
                return self.task("cifar10")

        def task_iris(self):
                return self.task("iris")

        def task_wine(self):
                return self.task("wine")

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return {"task": "synth-charset", "type": ctype, "color": color, "irows": irows, "icols": icols, "count": count}

        def task_synth_nparity(self, n = 32, count = 10000):
                return {"task": "synth-nparity", "n": n, "count": count}

        def task_synth_affine(self, isize = 32, osize = 32, noise = 0.0, count = 10000):
                return {"task": "synth-affine", "isize": isize, "osize": osize, "noise": noise, "count": count}

        def task_synth_peak2d(self, irows = 32, icols = 32, noise = 0.0, count = 10000):
                return {"task": "synth-peak2d", "irows": irows, "icols": icols, "noise": noise, "count": count}
