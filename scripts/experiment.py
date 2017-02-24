import os
import re
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

        def add_trainers(self, pool, names):
                for name in names:
                        self.trainers[name] = pool.get(name)

        def add_losses(self, pool, names):
                for name in names:
                        self.losses[name] = pool.get(name)

        def add_criteria(self, pool, names):
                for name in names:
                        self.criteria[name] = pool.get(name)

        def get_path(self, trial, mname, tname, cname, lname, extension):
                return self.dir + "/trial" + str(trial) + "_" + tname + "_" + mname + "_" + cname + "_" + lname + extension

        def filter(self, trial, mname_reg, tname_reg, cname_reg, lname_reg, extension):
                paths = []
                for mname in self.models:
                        if not re.match(mname_reg, mname):
                                continue
                        for tname in self.trainers:
                                if not re.match(tname_reg, tname):
                                        continue
                                for cname in self.criteria:
                                        if not re.match(cname_reg, cname):
                                                continue
                                        for lname in self.losses:
                                                if not re.match(lname_reg, lname):
                                                        continue
                                                paths.append(self.get_path(trial, mname, tname, cname, lname, extension))
                return paths

        def train_one(self, param, lpath):
                lfile = open(lpath, "w")
                self.log("running <", param, ">...")
                print("running <", param, ">...", file = lfile)
                subprocess.check_call((self.app_train + " " + param).split(), stdout = lfile)
                lfile.close()
                self.log("|--->training done, see <", lpath, ">")

        def get_csv(self, spath):
                # state file with the following format:
                #  (epoch, {train, valid, test} x {criterion, loss{average, variance, maximum}, error{average, variance, maximum}}, time)+
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
                        for col in (0, 1, 4):
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
                        for col in (0, 1, 4, 7, 8, 11, 14, 15, 18):
                                # plot wrt epoch/iteration number
                                self.plot_many_wrt(spaths, names, datas, pdf, 0, col)
                                # plot wrt time
                                self.plot_many_wrt(spaths, names, datas, pdf, 22, col)
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

        def run_trial(self, trial, epochs, policy, min_batch, max_batch):
                for mname, mparam in self.models.items():
                        for tname, tparam in self.trainers.items():
                                tparam += ",epochs={0},policy={1},min_batch={2},max_batch={3}".format(epochs, policy, min_batch, max_batch)
                                for cname, cparam in self.criteria.items():
                                        for lname, lparam in self.losses.items():
                                                self.run_one(trial, mname, mparam, tname, tparam, cname, cparam, lname, lparam)

        def run_all(self, trials = 10, epochs = 1000, policy = "", min_batch = 32, max_batch = 256):
                for trial in range(trials):
                        self.run_trial(trial, epochs, policy, min_batch, max_batch)

        def get_token(self, line, begin_delim, end_delim, start = 0):
                begin = line.find(begin_delim, start) + len(begin_delim)
                end = line.find(end_delim, begin)
                return line[begin : end], end

        def get_seconds(self, delta):
                delta = delta.replace("ms", "/1000")
                delta = delta.replace("s:", "+")
                delta = delta.replace("m:", "*60*60+")
                delta = delta.replace("h:", "*60*60*60+")
                delta = delta.replace("d:", "*24*60*60*60+")
                if delta[0] == '0':
                        delta = delta[1:]
                delta = delta.replace("+00*", "1*")
                delta = delta.replace("+00", "")
                delta = delta.replace("+0", "+")
                return str(eval(delta))

        def get_log(self, lpath):
                lfile = open(lpath, "r")
                for line in lfile:
                        if line.find("speed=") < 0:
                                continue
                        # search for the test value (aka criterion)
                        value, index = self.get_token(line, "test=", "|", 0)
                        # search for the test error
                        error, index = self.get_token(line, "|", "+/-", index)
                        # search for the optimum number of epochs
                        epoch, index = self.get_token(line, "epoch=", ",", index)
                        # search for the convergence speed
                        speed, index = self.get_token(line, "speed=", ",", index)
                        # duration
                        delta, index = self.get_token(line, "time=", ".", index)
                        return value, error, epoch, speed, self.get_seconds(delta)
                lfile.close()
                print("invalid log file <", lpath, ">")

        def summarize_one(self, trials, mname, tname, cname, lname, lfile):
                cmdline = self.app_stats + " -p 4"
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(trials):
                        lpath = self.get_path(trial, mname, tname, cname, lname, ".log")
                        value, error, epoch, speed, delta = self.get_log(lpath)
                        values.append(value)
                        errors.append(error)
                        epochs.append(epoch)
                        speeds.append(speed)
                        deltas.append(delta)
                value_stats = subprocess.check_output(cmdline.split() + values).decode('utf-8').strip()
                error_stats = subprocess.check_output(cmdline.split() + errors).decode('utf-8').strip()
                epoch_stats = subprocess.check_output(cmdline.split() + epochs).decode('utf-8').strip()
                speed_stats = subprocess.check_output(cmdline.split() + speeds).decode('utf-8').strip()
                delta_stats = subprocess.check_output(cmdline.split() + deltas).decode('utf-8').strip()
                print("%-12s | %-14s | %-16s | %-16s | %-34s | %-34s | %-48s | %-34s | %-48s" % \
                        (mname, tname, cname, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats),
                        file = lfile)

        def summarize(self, trials):
                lpath = self.dir + "/result.log"
                lfile = open(lpath, "w")

                # header
                print("%-12s | %-14s | %-16s | %-16s | %-34s | %-34s | %-48s | %-34s | %-48s" % \
                        ("model", "trainer", "criterion", "loss", "test value", "test error", "epochs", "convergence speed", "duration (sec)"),
                        file = lfile)
                print("-" * 280, file = lfile)
                # content
                for mname in self.models:
                        for tname in self.trainers:
                                for cname in self.criteria:
                                        for lname in self.losses:
                                                self.summarize_one(trials, mname, tname, cname, lname, lfile)
                lfile.close()

                # print file to screen
                lfile = open(lpath, "r")
                self.log()
                print(lfile.read())
                lfile.close()
