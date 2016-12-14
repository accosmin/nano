import os
import subprocess
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

        def train(self, param, lpath):
                lfile = open(lpath, "a")
                subprocess.call((self.trainer + " " + param).split(), stdout = lfile)
                lfile.close()

        def plot(self, spath, ppath):
                # state file with the following format:
                #  ({train, valid, test} x {criterion, loss{average, variance, maximum}, error{average, variance, maximum}}, time)+
                data = mlab.csv2rec(spath, delimiter = ' ', names = None)
                with PdfPages(ppath) as pdf:
                        for col in range(7):
                                # x axis - epoch/iteration index
                                xname = data.dtype.names[0]
                                # y axis - train/validation/test datasets
                                yname0 = data.dtype.names[col + 1]
                                yname1 = data.dtype.names[col + 8]
                                yname2 = data.dtype.names[col + 15]
                                # plot
                                plt.xlabel(xname)
                                plt.ylabel(yname0.strip("train_"))
                                plt.title(os.path.basename(spath).strip(".state"))
                                plt.plot(data[xname], data[yname0], "r--", label = yname0)
                                plt.plot(data[xname], data[yname1], "g-.", label = yname1)
                                plt.plot(data[xname], data[yname2], "b-o", label = yname2)
                                plt.legend()
                                pdf.savefig()
                                plt.close()

        def run_one(self, trial, mname, mparam, tname, tparam, cname, cparam, lname, lparam):
                os.makedirs(self.dir, exist_ok = True)
                mpath = self.make_path(trial, mname, tname, cname, lname, ".model")
                spath = self.make_path(trial, mname, tname, cname, lname, ".state")
                lpath = self.make_path(trial, mname, tname, cname, lname, ".log")
                ppath = self.make_path(trial, mname, tname, cname, lname, ".pdf")

                param = self.task + " " + mparam + " " + tparam + " " + cparam + " " + lparam + " --model-file " + mpath
                print("running <", param, ">...")
                lfile = open(lpath, "w")
                print("running <", param, ">...", file = lfile)
                lfile.close()
                self.train(param, lpath)
                print("|--->training done, see <", lpath, ">")
                self.plot(spath, ppath)
                print("|--->plotting done, see <", ppath, ">")
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
