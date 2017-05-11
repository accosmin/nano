import os
import re
import time
import config
import plotter
import subprocess
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
from matplotlib.backends.backend_pdf import PdfPages

class experiment:
        def __init__(self, task, outdir):
                self.cfg = config.config()
                self.task = task
                self.dir = outdir
                self.models = SortedDict({})
                self.trainers = SortedDict({})
                self.iterators = SortedDict({})
                self.losses = SortedDict({})

        def log(self, *messages):
                print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

        def add_model(self, name, params):
                self.models[name] = params

        def add_trainers(self, names):
                for name in names:
                        self.trainers[name] = self.cfg.trainers().get(name)

        def add_losses(self, names):
                for name in names:
                        self.losses[name] = self.cfg.losses().get(name)

        def add_iterators(self, names):
                for name in names:
                        self.iterators[name] = self.cfg.iterators().get(name)

        def get_path(self, trial, mname, tname, iname, lname, extension):
                return self.dir + "/trial" + str(trial) + "_" + tname + "_" + mname + "_" + iname + "_" + lname + extension

        def filter(self, trial, mname_reg, tname_reg, iname_reg, lname_reg, extension):
                paths = []
                for mname in self.models:
                        if not re.match(mname_reg, mname):
                                continue
                        for tname in self.trainers:
                                if not re.match(tname_reg, tname):
                                        continue
                                for iname in self.iterators:
                                        if not re.match(iname_reg, iname):
                                                continue
                                        for lname in self.losses:
                                                if not re.match(lname_reg, lname):
                                                        continue
                                                paths.append(self.get_path(trial, mname, tname, iname, lname, extension))
                return paths

        def train_one(self, param, lpath):
                lfile = open(lpath, "w")
                self.log("running <", param, ">...")
                print("running <", param, ">...", file = lfile)
                subprocess.check_call((self.cfg.app_train + " " + param).split(), stdout = lfile)
                lfile.close()
                self.log("|--->training done, see <", lpath, ">")

        def plot_one(self, spath, ppath):
                plotter.plot_one(spath, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_many(self, spaths, ppath):
                plotter.plot_many(spaths, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def run_one(self, trial, mname, mparam, tname, tparam, iname, iparam, lname, lparam):
                os.makedirs(self.dir, exist_ok = True)
                mpath = self.get_path(trial, mname, tname, iname, lname, ".model")
                spath = self.get_path(trial, mname, tname, iname, lname, ".state")
                lpath = self.get_path(trial, mname, tname, iname, lname, ".log")
                ppath = self.get_path(trial, mname, tname, iname, lname, ".pdf")

                param = self.task + " " + mparam + " " + tparam + " " + iparam + " " + lparam + " --model-file " + mpath
                self.train_one(param, lpath)
                self.plot_one(spath, ppath)
                self.log()

        def run_trial(self, trial, epochs, policy, min_batch, max_batch, patience, epsilon):
                for mname, mparam in self.models.items():
                        for tname, tparam in self.trainers.items():
                                tformat = ",epochs={0},policy={1},min_batch={2},max_batch={3},patience={4},eps={5}"
                                tparam += tformat.format(epochs, policy, min_batch, max_batch, patience, epsilon)
                                for iname, iparam in self.iterators.items():
                                        for lname, lparam in self.losses.items():
                                                self.run_one(trial, mname, mparam, tname, tparam, iname, iparam, lname, lparam)

        def run_all(self, trials = 10, epochs = 1000, policy = "stop_early", min_batch = 32, max_batch = 256, patience = 32, epsilon = 1e-6):
                for trial in range(trials):
                        self.run_trial(trial, epochs, policy, min_batch, max_batch, patience, epsilon)

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
                while (len(delta) > 0) and (delta[0] == '0'):
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
                        # search for the test loss value
                        value, index = self.get_token(line, "test=", "|", 0)
                        # search for the test error
                        error, index = self.get_token(line, "|", ",", index)
                        # search for the optimum number of epochs
                        epoch, index = self.get_token(line, "epoch=", ",", index)
                        # search for the convergence speed
                        speed, index = self.get_token(line, "speed=", "/s", index)
                        # duration
                        delta, index = self.get_token(line, "time=", ".", index)
                        return value, error, epoch, speed, self.get_seconds(delta)
                lfile.close()
                print("invalid log file <", lpath, ">")

        def summarize_one(self, trials, mname, tname, iname, lname, lfile):
                cmdline = self.cfg.app_stats + " -p 4"
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(trials):
                        lpath = self.get_path(trial, mname, tname, iname, lname, ".log")
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
                print("{0};{1};{2};{3};{4};{5};{6};{7};{8}".format(
                        mname, tname, iname, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats),
                        file = lfile)

        def summarize(self, trials):
                lpath = self.dir + "/result.log"
                cpath = self.dir + "/result.csv"

                cfile = open(cpath, "w")
                # header
                print("{0};{1};{2};{3};{4};{5};{6};{7};{8}".format(
                        "model", "trainer", "iterator", "loss", "test value", "test error", "epochs", "convergence speed", "duration (sec)"),
                        file = cfile)
                # content
                for mname in self.models:
                        for tname in self.trainers:
                                for iname in self.iterators:
                                        for lname in self.losses:
                                                self.summarize_one(trials, mname, tname, iname, lname, cfile)
                cfile.close()

                # tabulate
                lfile = open(lpath, "w")
                subprocess.check_call((self.cfg.app_tabulate + " -i " + cpath + " -d \";\"").split(), stdout = lfile)
                lfile.close()

                # print file to screen
                lfile = open(lpath, "r")
                self.log()
                print(lfile.read())
                lfile.close()
