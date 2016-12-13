import os
import subprocess

class experiment:
        def __init__(self, trainer, task, outdir):
                self.trainer = trainer
                self.task = task
                self.dir = outdir
                self.models = {}
                self.trainers = {}
                self.criteria = {}
                self.losses = {}

        def add_model(self, name, params):
                self.models[name] = params

        def add_trainer(self, name, params):
                self.trainers[name] = params

        def add_loss(self, name, params):
                self.losses[name] = params

        def add_criterion(self, name, params):
                self.criteria[name] = params

        def make_path(self, trial, mname, tname, cname, lname, extension):
                return self.dir + "/trial" + str(trial) + "_" + tname + "_" + mname + "_" + cname + "_" + lname + extension

        def run_one(self, trial, mname, mparam, tname, tparam, cname, cparam, lname, lparam):
                os.makedirs(self.dir, exist_ok = True)
                mpath = self.make_path(trial, mname, tname, cname, lname, ".model")
                spath = self.make_path(trial, mname, tname, cname, lname, ".state")
                lpath = self.make_path(trial, mname, tname, cname, lname, ".log")
                ppath = self.make_path(trial, mname, tname, cname, lname, ".pdf")

                lfile = open(lpath, "wt")

                param = self.task + " " + mparam + " " + tparam + " " + cparam + " " + lparam + " --model-file " + mpath
                print("running <", param, ">...")
                print("running <", param, ">...", file = lfile)
                subprocess.call((self.trainer + " " + param).split(), stdout = lfile)
                print("  training done, see <", lpath, ">")
                #bash $(dirname $0)/plot_model.sh ${sfile}
                print("  plotting done, see <", ppath, ">")
                print()

        def run_trial(self, trial, epochs, policy):
                for mname, mparam in self.models.items():
                        for tname, tparam in self.trainers.items():
                                tparam += ",epochs=" + str(epochs) + str(policy)
                                for cname, cparam in self.criteria.items():
                                        for lname, lparam in self.losses.items():
                                                self.run_one(trial, mname, mparam, tname, tparam, cname, cparam, lname, lparam)

        def run_all(self, trials, epochs, policy):
                for trial in range(trials):
                        self.run_trial(trial, epochs, policy)
