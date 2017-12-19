import os
import json

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

        # available losses
        def losses(self):
                return ["cauchy",
                        "square",
                        "classnll",
                        "s-logistic",
                        "m-logistic",
                        "s-exponential",
                        "m-exponential"]

        # available enhancers
        def enhancers(self):
                return ["default",
                        "noise",
                        "warp",
                        "noclass"]

                        "noclass" : "--enhancer noclass"
                }

        # available stochastic solvers
        def stoch_solvers(self):
                return ["ag", "agfr", "aggr",
                        "sg", "sgm", "ngd", "svrg", "asgd",
                        "adagrad", "adadelta", "adam", "rmsprop"]

        # available batch solvers
        def batch_solvers(self):
                return ["gd", "cgd", "lbfgs"]

        # configure stochastic trainer
        def trainer_stoch(self, solver, epochs = 100, patience = 32, epsilon = 1e-6, batch = 32):
                return json.dumps({
                        "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon, "batch": batch})

        # configure batch trainer
        def trainer_batch(self, solver, epochs = 100, patience = 32, epsilon = 1e-6):
                return json.dumps({
                        "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon})

        # configure tasks
        def task(self, name):
                return json.dumps({"task": name, "dir": os.path.join(self.dbdir, name)})

        def task_mnist(self):
                return self.task("mnist")

        def task_cifar10(self):
                return self.task("cifar10")

        def task_iris(self):
                return self.task("iris")

        def task_wine(self):
                return self.task("wine")

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return json.dumps({"task": "synth-charset", "type": ctype, "color": color, "irows": irows, "icols": icols, "count": count})

        def task_synth_nparity(self, n = 32, count = 10000):
                return json.dumps({"task": "synth-nparity", "n": n, "count": count})
