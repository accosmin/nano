import os
import re
import time
import config
import plotter
import subprocess
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class experiment:
        def __init__(self, task, outdir):
                self.cfg = config.config()
                self.task = task
                self.dir = outdir
                self.models = []
                self.trainers = []
                self.iterators = []
                self.losses = []

        def log(self, *messages):
                print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

        def add_model(self, name, parameters):
                self.models.append([name, parameters])

        def add_trainer(self, name, parameters, config_name = None):
                self.trainers.append([config_name if config_name else name, self.cfg.config_trainer(name, parameters)])

        def add_loss(self, name, parameters = "", config_name = None):
                self.losses.append([config_name if config_name else name, self.cfg.config_loss(name, parameters)])

        def add_iterator(self, name, parameters = "", config_name = None):
                self.iterators.append([config_name if config_name else name, self.cfg.config_iterator(name, parameters)])

        def path(self, trial, mname, tname, iname, lname, extension):
                basepath = self.dir
                basepath += "trial{}".format(trial) if not (trial is None) else "result"
                basepath += "_" + mname if mname else ""
                basepath += "_" + tname if tname else ""
                basepath += "_" + iname if iname else ""
                basepath += "_" + lname if lname else ""
                basepath += extension
                return basepath

        def name(self, name_config):
                return name_config[0]

        def config(self, name_config):
                return name_config[1]

        def names(self, names_configs):
                names = []
                for name_config in names_configs:
                        names.append(self.name(name_config))
                return names

        def filter(self, trial, mname_reg, tname_reg, iname_reg, lname_reg, extension):
                paths = []
                for mname in self.names(self.models):
                        if not re.match(mname_reg, mname):
                                continue
                        for tname in self.names(self.trainers):
                                if not re.match(tname_reg, tname):
                                        continue
                                for iname in self.names(self.iterators):
                                        if not re.match(iname_reg, iname):
                                                continue
                                        for lname in self.names(self.losses):
                                                if not re.match(lname_reg, lname):
                                                        continue
                                                paths.append(self.path(trial, mname, tname, iname, lname, extension))
                return paths

        def train_one(self, param, lpath):
                with open(lpath, "w") as lfile:
                        self.log("running <", param, ">...")
                        subprocess.check_call((self.cfg.app_train + " " + param).split(), stdout = lfile)
                self.log("|--->training done, see <", lpath, ">")

        def plot_one(self, spath, ppath):
                plotter.plot_state_one(spath, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_many(self, spaths, ppath):
                plotter.plot_state_many(spaths, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_trial(self, spaths, ppath, names):
                plotter.plot_trial_many(spaths, ppath, names)
                self.log("|--->plotting done, see <", ppath, ">")

        def run_one(self, trial, mname, mparam, tname, tparam, iname, iparam, lname, lparam):
                os.makedirs(self.dir, exist_ok = True)
                mpath = self.path(trial, mname, tname, iname, lname, ".model")
                spath = self.path(trial, mname, tname, iname, lname, ".state")
                lpath = self.path(trial, mname, tname, iname, lname, ".log")
                ppath = self.path(trial, mname, tname, iname, lname, ".pdf")

                param = self.task + " " + mparam + " " + tparam + " " + iparam + " " + lparam + " --model-file " + mpath
                self.train_one(param, lpath)
                self.plot_one(spath, ppath)
                self.log()

        def run_trials(self, trials, mname, mparam, tname, tparam, iname, iparam, lname, lparam):
                for trial in range(trials):
                        self.run_one(trial, mname, mparam, tname, tparam, iname, iparam, lname, lparam)

                # export the results from multiple trials as csv
                cpath = self.path(None, mname, tname, iname, lname, ".csv")
                with open(cpath, "w") as cfile:
                        print(self.get_csv_header(), file = cfile)
                        values, errors, epochs, speeds, deltas = self.get_logs(trials, mname, tname, iname, lname)
                        for value, error, epoch, speed, delta in zip(values, errors, epochs, speeds, deltas):
                                print(self.get_csv_row(mname, tname, iname, lname, value, error, epoch, speed, delta), file = cfile)

        def run_all(self, trials = 10):
                for mdata in self.models:
                        for tdata in self.trainers:
                                for idata in self.iterators:
                                        for ldata in self.losses:
                                                self.run_trials(trials,
                                                        self.name(mdata), self.config(mdata),
                                                        self.name(tdata), self.config(tdata),
                                                        self.name(idata), self.config(idata),
                                                        self.name(ldata), self.config(ldata))

        def summarize_by_models(self, trials):
                tnames = self.names(self.trainers)
                inames = self.names(self.iterators)
                lnames = self.names(self.losses)
                for tname, iname, lname in [(x, y, z) for x in tnames for y in inames for z in lnames]:
                        for trial in range(trials):
                                self.plot_many(
                                        self.filter(trial, ".*", tname, iname, lname, ".state"),
                                        self.path(trial, None, tname, iname, lname, ".pdf"))

                        self.summarize(trials, ".*", tname, iname, lname,
                                self.path(None, None, tname, iname, lname, ".log"),
                                self.path(None, None, tname, iname, lname, ".csv"),
                                self.path(None, None, tname, iname, lname, ".pdf"),
                                self.names(self.models))

        def summarize_by_trainers(self, trials):
                mnames = self.names(self.models)
                inames = self.names(self.iterators)
                lnames = self.names(self.losses)
                for mname, iname, lname in [(x, y, z) for x in mnames for y in inames for z in lnames]:
                        for trial in range(trials):
                                self.plot_many(
                                        self.filter(trial, mname, ".*", iname, lname, ".state"),
                                        self.path(trial, mname, None, iname, lname, ".pdf"))

                        self.summarize(trials, mname, ".*", iname, lname,
                                self.path(None, mname, None, iname, lname, ".log"),
                                self.path(None, mname, None, iname, lname, ".csv"),
                                self.path(None, mname, None, iname, lname, ".pdf"),
                                self.names(self.trainers))

        def summarize_by_iterators(self, trials):
                mnames = self.names(self.models)
                tnames = self.names(self.trainers)
                lnames = self.names(self.losses)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        for trial in range(trials):
                                self.plot_many(
                                        self.filter(trial, mname, tname, ".*", lname, ".state"),
                                        self.path(trial, mname, tname, None, lname, ".pdf"))

                        self.summarize(trials, mname, tname, ".*", lname,
                                self.path(None, mname, tname, None, lname, ".log"),
                                self.path(None, mname, tname, None, lname, ".csv"),
                                self.path(None, mname, tname, None, lname, ".pdf"),
                                self.names(self.iterators))

        def summarize_by_losses(self, trials):
                mnames = self.names(self.models)
                tnames = self.names(self.trainers)
                inames = self.names(self.iterators)
                for mname, tname, iname in [(x, y, z) for x in mnames for y in tnames for z in inames]:
                        for trial in range(trials):
                                self.plot_many(
                                        self.filter(trial, mname, tname, iname, ".*", ".state"),
                                        self.path(trial, mname, tname, iname, None, ".pdf"))

                        self.summarize(trials, mname, tname, iname, ".*",
                                self.path(None, mname, tname, iname, None, ".log"),
                                self.path(None, mname, tname, iname, None, ".csv"),
                                self.path(None, mname, tname, iname, None, ".pdf"),
                                self.names(self.losses))

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
                with open(lpath, "r") as lfile:
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
                print("invalid log file <", lpath, ">")

        def get_logs(self, trials, mname, tname, iname, lname):
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(trials):
                        lpath = self.path(trial, mname, tname, iname, lname, ".log")
                        value, error, epoch, speed, delta = self.get_log(lpath)
                        values.append(value)
                        errors.append(error)
                        epochs.append(epoch)
                        speeds.append(speed)
                        deltas.append(delta)
                return values, errors, epochs, speeds, deltas

        def get_csv_header(self, delim = ";"):
                return delim.join(["model", "trainer", "iterator", "loss", "test value", "test error", "epochs", "convergence speed", "duration (sec)"])

        def get_csv_row(self, mname, tname, iname, lname, value, error, epoch, speed, delta, delim = ";"):
                return delim.join([mname, tname, iname, lname, value, error, epoch, speed, delta])

        def summarize_one(self, trials, mname, tname, iname, lname, cfile):
                values, errors, epochs, speeds, deltas = self.get_logs(trials, mname, tname, iname, lname)
                cmdline = self.cfg.app_stats + " -p 4"
                value_stats = subprocess.check_output(cmdline.split() + values).decode('utf-8').strip()
                error_stats = subprocess.check_output(cmdline.split() + errors).decode('utf-8').strip()
                epoch_stats = subprocess.check_output(cmdline.split() + epochs).decode('utf-8').strip()
                speed_stats = subprocess.check_output(cmdline.split() + speeds).decode('utf-8').strip()
                delta_stats = subprocess.check_output(cmdline.split() + deltas).decode('utf-8').strip()
                print(self.get_csv_row(mname, tname, iname, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats), file = cfile)

        def summarize(self, trials, mname_reg, tname_reg, iname_reg, lname_reg, lpath, cpath, ppath, names):
                # plot models
                self.plot_trial(
                        self.filter(None, mname_reg, tname_reg, iname_reg, lname_reg, ".csv"),
                        ppath, names)

                # export the results as csv
                with open(cpath, "w") as cfile:
                        print(self.get_csv_header(), file = cfile)
                        for mname in self.names(self.models):
                                if not re.match(mname_reg, mname):
                                        continue
                                for tname in self.names(self.trainers):
                                        if not re.match(tname_reg, tname):
                                                continue
                                        for iname in self.names(self.iterators):
                                                if not re.match(iname_reg, iname):
                                                        continue
                                                for lname in self.names(self.losses):
                                                        if not re.match(lname_reg, lname):
                                                                continue
                                                        self.summarize_one(trials, mname, tname, iname, lname, cfile)

                # tabulate
                with open(lpath, "w") as lfile:
                        subprocess.check_call((self.cfg.app_tabulate + " -i " + cpath + " -d \";\"").split(), stdout = lfile)

                # print file to screen
                with open(lpath, "r") as lfile:
                        self.log()
                        print(lfile.read())
