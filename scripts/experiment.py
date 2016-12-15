import os
import time
import subprocess
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class experiment:
        def __init__(self, app_train, app_stats, task, outdir):
                self.app_train = app_train
                self.app_stats = app_stats
                self.task = task
                self.dir = outdir
                self.models = {}
                self.trainers = {}
                self.criteria = {}
                self.losses = {}

        def log(self, *messages):
                print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

        def add_model(self, name, params):
                self.models[name] = params

        def add_trainer(self, name, params):
                self.trainers[name] = params

        def add_loss(self, name, params):
                self.losses[name] = params

        def add_criterion(self, name, params):
                self.criteria[name] = params

        def get_path(self, trial, mname, tname, cname, lname, extension):
                return self.dir + "/trial" + str(trial) + "_" + tname + "_" + mname + "_" + cname + "_" + lname + extension

        def train_one(self, param, lpath):
                lfile = open(lpath, "w")
                self.log("running <", param, ">...")
                print("running <", param, ">...", file = lfile)
                subprocess.check_call((self.app_train + " " + param).split(), stdout = lfile)
                lfile.close()
                self.log("|--->training done, see <", lpath, ">")

        def get_csv(self, spath):
                # state file with the following format:
                #  ({train, valid, test} x {criterion, loss{average, variance, maximum}, error{average, variance, maximum}}, time)+
                name = os.path.basename(spath).replace(".state", "")
                name = name.replace(name[0 : name.find("_") + 1], "")
                data = mlab.csv2rec(spath, delimiter = ' ', names = None)
                return name, data

        def get_csvs(self, spaths):
                datas = []
                names = []
                for spath in spaths:
                        name, data = self.get_csv(spath)
                        datas.append(data)
                        names.append(name)
                return names, datas

        def plot_one(self, spath, ppath):
                title, data = self.get_csv(spath)
                with PdfPages(ppath) as pdf:
                        for col in range(7):
                                # x axis - epoch/iteration index
                                xname = data.dtype.names[0]
                                xlabel = xname
                                # y axis - train/validation/test datasets
                                yname0 = data.dtype.names[col + 1]
                                yname1 = data.dtype.names[col + 8]
                                yname2 = data.dtype.names[col + 15]
                                ylabel = yname0.replace("train_", "")
                                # plot
                                plt.xlabel(xlabel, fontsize = "smaller")
                                plt.ylabel(ylabel, fontsize = "smaller")
                                plt.title(title, weight = "bold")
                                plt.plot(data[xname], data[yname0], "r--", label = yname0)
                                plt.plot(data[xname], data[yname1], "g:", label = yname1)
                                plt.plot(data[xname], data[yname2], "b-", label = yname2)
                                plt.legend(fontsize = "smaller")
                                pdf.savefig()
                                plt.close()
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_many_wrt(self, spaths, names, datas, pdf, xcol, ycol):
                colnames = datas[0].dtype.names
                title = colnames[ycol + 1]
                # x axis - epoch/iteration index
                xname = colnames[xcol]
                xlabel = xname
                # y axis - train/validation/test datasets
                yname = colnames[ycol + 1]
                ylabel = yname.replace("train_", "").replace("valid_", "").replace("test_", "")
                # plot
                plt.xlabel(xlabel, fontsize = "smaller")
                plt.ylabel(ylabel, fontsize = "smaller")
                plt.title(title, weight = "bold")
                for i, spath in enumerate(spaths):
                        plt.plot(datas[i][xname], datas[i][yname], label = names[i])
                plt.legend(fontsize = "smaller")
                pdf.savefig()
                plt.close()

        def plot_many(self, spaths, ppath):
                names, datas = self.get_csvs(spaths)
                with PdfPages(ppath) as pdf:
                        for col in range(21):
                                # plot wrt epoch/iteration number
                                self.plot_many_wrt(spaths, names, datas, pdf, 0, col + 1)
                                # plot wrt time
                                self.plot_many_wrt(spaths, names, datas, pdf, 22, col + 1)
                self.log("|--->plotting done, see <", ppath, ">")

        def run_one(self, trial, mname, mparam, tname, tparam, cname, cparam, lname, lparam):
                os.makedirs(self.dir, exist_ok = True)
                mpath = self.get_path(trial, mname, tname, cname, lname, ".model")
                spath = self.get_path(trial, mname, tname, cname, lname, ".state")
                lpath = self.get_path(trial, mname, tname, cname, lname, ".log")
                ppath = self.get_path(trial, mname, tname, cname, lname, ".pdf")

                param = self.task + " " + mparam + " " + tparam + " " + cparam + " " + lparam + " --model-file " + mpath
                self.train_one(param, lpath)
                self.plot_one(spath, ppath)
                self.log()

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

        def get_value(self, line, begin_delim, end_delim, start = 0):
                begin = line.find(begin_delim, start)
                if begin < 0:
                        return "", 0
                begin += len(begin_delim)
                end = line.find(end_delim, begin)
                if end < 0:
                        return "", 0
                return line[begin : end], end

        def get_log(self, lpath):
                lfile = open(lpath, "r")
                for line in lfile:
                        # search for the test value (aka criterion)
                        value, index = self.get_value(line, "test=", "|", 0)
                        if value:
                                # search for the test error
                                error, index = self.get_value(line, "|", "+/-", index)
                                if error:
                                        # search for the optimum number of epochs
                                        epoch, index = self.get_value(line, "epoch=", ",", index)
                                        if epoch:
                                                return value, error, epoch
                lfile.close()
                return 0, 0, 0

        def summarize_one(self, trials, mname, tname, cname, lname, lfile):
                values = []
                errors = []
                epochs = []
                for trial in range(trials):
                        lpath = self.get_path(trial, mname, tname, cname, lname, ".log")
                        value, error, epoch = self.get_log(lpath)
                        values.append(value)
                        errors.append(error)
                        epochs.append(epoch)
                value_stats = subprocess.check_output(self.app_stats.split() + values).decode('utf-8').strip()
                error_stats = subprocess.check_output(self.app_stats.split() + errors).decode('utf-8').strip()
                epoch_stats = subprocess.check_output(self.app_stats.split() + epochs).decode('utf-8').strip()
                print("%-16s %-16s %-16s %-16s %-48s %-48s %-48s" % \
                        (mname, tname, cname, lname, value_stats, error_stats, epoch_stats), file = lfile)

        def summarize(self, trials):
                lpath = self.dir + "/result.log"
                lfile = open(lpath, "w")

                print("%-16s %-16s %-16s %-16s %-48s %-48s %-48s" % \
                        ("model", "trainer", "criterion", "loss", "test value", "test error", "epochs"), file = lfile)
                print("-" * 220, file = lfile)
                print(file = lfile)

                for mname in self.models:
                        for tname in self.trainers:
                                for cname in self.criteria:
                                        for lname in self.losses:
                                                self.summarize_one(trials, mname, tname, cname, lname, lfile)

                lfile.close()

                # print file to screen
                lfile = open(lpath, "r")
                self.log(lfile.read())
                lfile.close()
