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
                        "s-cauchy",             # classification (single label)
                        "m-cauchy",             # classification (multi label)
                        "square",               # regression
                        "s-square",             # classification (single label)
                        "m-square",             # classification (multi label)
                        "classnll",             # classification (single label)
                        "s-logistic",           # classification (single label)
                        "m-logistic",           # classification (multi label)
                        "s-exponential",        # classification (single label)
                        "m-exponential"]        # classification (multi label)

        def solvers(self):
                """ available batch (line-search) solvers """
                return ["gd", "cgd", "lbfgs", "bfgs"]

        def loss(self, loss):
                """ create a loss """
                assert(loss in self.losses())
                return {"id": loss}

        def task(self, name, folds):
                """ create a task """
                return {"id": name, "dir": os.path.join(self.dbdir, name), "folds": folds}

        def task_mnist(self, folds):
                return self.task("mnist", folds)

        def task_cifar10(self, folds):
                return self.task("cifar10", folds)

        def task_iris(self, folds):
                return self.task("iris", folds)

        def task_wine(self, folds):
                return self.task("wine", folds)
