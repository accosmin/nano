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
        """ an experiment run consists of:
                - a trial (fold index or None if cumulating results from multiple trials)
                - a model (name + command line parameters to apps/builder)
                - a trainer (name + json parameters)
                - an enhancer (name + json parameters)
                - a loss function (name)
        """
        def __init__(self, task, outdir, trials = 10):
                self.cfg = config.config()
                self.task = task
                self.dir = outdir
                self.trials = trials
                self.runs = []
                self.losses = []
                self.models = []
                self.trainers = []
                self.enhancers = []

        def log(self, *messages):
                print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

        def add_model(self, name, parameters):
                self.models.append([name, parameters])

        def add_trainer(self, name, parameters, config_name = None):
                self.trainers.append([config_name if config_name else name, parameters])

        def add_loss(self, name, parameters = "", config_name = None):
                self.losses.append([config_name if config_name else name, parameters])

        def add_enhancer(self, name, parameters = "", config_name = None):
                self.enhancers.append([config_name if config_name else name, parameters])

        def path(self, trial, mname, tname, ename, lname, extension):
                basepath = self.dir
                basepath += "trial{}".format(trial) if not (trial is None) else "result"
                basepath += "_M" + mname if mname else ""
                basepath += "_T" + tname if tname else ""
                basepath += "_E" + ename if ename else ""
                basepath += "_L" + lname if lname else ""
                basepath += extension
                return basepath

        def name(self, name_config):
                return name_config[0]

        def config(self, name_config):
                return name_config[1]

        def names(self, names_configs, name_reg = None):
                names = []
                for name_config in names_configs:
                        name = self.name(name_config)
                        if name_reg == None or re.match(name_reg, name):
                                names.append(name)
                return names

        def filter_names(self, mname_reg, tname_reg, ename_reg, lname_reg):
                mnames = self.names(self.models, mname_reg)
                tnames = self.names(self.trainers, tname_reg)
                enames = self.names(self.enhancers, ename_reg)
                lnames = self.names(self.losses, lname_reg)
                return mnames, tnames, enames, lnames

        def filter_paths(self, trial, mname_reg, tname_reg, ename_reg, lname_reg, extension):
                paths = []
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, tname_reg, ename_reg, lname_reg)
                for mname, tname, ename, lname in [(u, x, y, z) for u in mnames for x in tnames for y in enames for z in lnames]:
                        paths.append(self.path(trial, mname, tname, ename, lname, extension))
                return paths

        def reg2str(self, reg):
                return reg.replace(".*", "all").replace("*", "").replace(".", "") if reg else "all"

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

        def tabulate(self, cpath, lpath):
                with open(lpath, "w") as lfile:
                        subprocess.check_call((self.cfg.app_tabulate + " -i " + cpath + " -d \";\"").split(), stdout = lfile)

                with open(lpath, "r") as lfile:
                        self.log()
                        print(lfile.read())

        def summarize_by_models(self, mname_reg = ".*"):
                mname = self.reg2str(mname_reg)

        def train_trials(self, mname, mparam, tname, tparam, ename, eparam, lname, lparam):
                # train the given configuration for multiple trials
                for trial in range(self.trials):
                        os.makedirs(self.dir, exist_ok = True)
                        mpath = self.path(trial, mname, tname, ename, lname, ".model")
                        spath = self.path(trial, mname, tname, ename, lname, ".state")
                        lpath = self.path(trial, mname, tname, ename, lname, ".log")
                        ppath = self.path(trial, mname, tname, ename, lname, ".pdf")

                        param = self.task + " " + mparam + " " + tparam + " " + eparam + " " + lparam + " --model-file " + mpath
                        self.train_one(param, lpath)
                        self.plot_one(spath, ppath)
                        self.log()

                # export the results from multiple trials as csv
                cpath = self.path(None, mname, tname, ename, lname, ".csv")
                spath = self.path(None, mname, tname, ename, lname, ".stats")
                with open(cpath, "w") as cfile, open(spath, "w") as sfile:
                        print(self.get_csv_header(), file = cfile)
                        print(self.get_csv_header(), file = sfile)
                        values, errors, epochs, speeds, deltas = self.get_logs( mname, tname, ename, lname)
                        for value, error, epoch, speed, delta in zip(values, errors, epochs, speeds, deltas):
                                print(self.get_csv_row(mname, tname, ename, lname, value, error, epoch, speed, delta), file = cfile)
                        value_stats, error_stats, epoch_stats, speed_stats, delta_stats = self.get_log_stats(values, errors, epochs, speeds, deltas)
                        print(self.get_csv_row(mname, tname, ename, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats), file = sfile)

                # display basic statistics for the results from multiple trials
                lpath = self.path(None, mname, tname, ename, lname, ".log")
                self.tabulate(spath, lpath)

        def train_all(self):
                # run all possible configurations
                mdatas = self.models
                tdatas = self.trainers
                edatas = self.enhancers
                ldatas = self.losses
                for mdata, tdata, edata, ldata in [(u, x, y, z) for u in mdatas for x in tdatas for y in edatas for z in ldatas]:
                        self.train_trials(
                                self.name(mdata), self.config(mdata),
                                self.name(tdata), self.config(tdata),
                                self.name(edata), self.config(edata),
                                self.name(ldata), self.config(ldata))

        def summarize_trials(self, mname, mname_reg, tname, tname_reg, ename, ename_reg, lname, lname_reg, names):
                # compare configurations for each trial: plot the training state evaluation
                for trial in range(self.trials):
                        self.plot_many(
                                self.filter_paths(trial, mname_reg, tname_reg, ename_reg, lname_reg, ".state"),
                                self.path(trial, mname, tname, ename, lname, ".pdf"))

                # compare configurations across all trials: boxplot the results
                ppath = self.path(None, mname, tname, ename, lname, ".pdf")

                self.plot_trial(
                        self.filter_paths(None, mname_reg, tname_reg, ename_reg, lname_reg, ".csv"),
                        ppath, names)

                # compare configurations across all trials: display basic statistics
                spath = self.path(None, mname, tname, ename, lname, ".stats")
                with open(spath, "w") as sfile:
                        print(self.get_csv_header(), file = sfile)
                        mnames, tnames, enames, lnames = self.filter_names(mname_reg, tname_reg, ename_reg, lname_reg)
                        for mname, tname, ename, lname in [(u, x, y, z) for u in mnames for x in tnames for y in enames for z in lnames]:
                                values, errors, epochs, speeds, deltas = self.get_logs(mname, tname, ename, lname)
                                value_stats, error_stats, epoch_stats, speed_stats, delta_stats = self.get_log_stats(values, errors, epochs, speeds, deltas)
                                print(self.get_csv_row(mname, tname, ename, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats), file = sfile)

                lpath = self.path(None, mname, tname, ename, lname, ".log")
                self.tabulate(spath, lpath)

        def summarize_by_models(self, mname_reg = ".*"):
                mname = self.reg2str(mname_reg)
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, None, None, None)
                for tname, ename, lname in [(x, y, z) for x in tnames for y in enames for z in lnames]:
                        self.summarize_trials(mname, mname_reg, tname, tname, ename, ename, lname, lname, mnames)

        def summarize_by_trainers(self, tname_reg = ".*"):
                tname = self.reg2str(tname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, tname_reg, None, None)
                for mname, ename, lname in [(x, y, z) for x in mnames for y in enames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname_reg, ename, ename, lname, lname, tnames)

        def summarize_by_enhancers(self, ename_reg = ".*"):
                ename = self.reg2str(ename_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, ename_reg, None)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname, ename, ename_reg, lname, lname, enames)

        def summarize_by_losses(self, lname_reg = ".*"):
                lname = self.reg2str(lname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, None, lname_reg)
                for mname, tname, ename in [(x, y, z) for x in mnames for y in tnames for z in enames]:
                        self.summarize_trials(mname, mname, tname, tname, ename, ename, lname, lname_reg, lnames)

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

        def get_logs(self, mname, tname, ename, lname):
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(self.trials):
                        lpath = self.path(trial, mname, tname, ename, lname, ".log")
                        value, error, epoch, speed, delta = self.get_log(lpath)
                        values.append(value)
                        errors.append(error)
                        epochs.append(epoch)
                        speeds.append(speed)
                        deltas.append(delta)
                return values, errors, epochs, speeds, deltas

        def get_log_stats(self, values, errors, epochs, speeds, deltas):
                cmdline = self.cfg.app_stats + " -p 4"
                value_stats = subprocess.check_output(cmdline.split() + values).decode('utf-8').strip()
                error_stats = subprocess.check_output(cmdline.split() + errors).decode('utf-8').strip()
                epoch_stats = subprocess.check_output(cmdline.split() + epochs).decode('utf-8').strip()
                speed_stats = subprocess.check_output(cmdline.split() + speeds).decode('utf-8').strip()
                delta_stats = subprocess.check_output(cmdline.split() + deltas).decode('utf-8').strip()
                return value_stats, error_stats, epoch_stats, speed_stats, delta_stats

        def get_csv_header(self, delim = ";"):
                return delim.join(["model", "trainer", "enhancer", "loss", "test value", "test error", "epochs", "convergence speed", "duration (sec)"])

        def get_csv_row(self, mname, tname, ename, lname, value, error, epoch, speed, delta, delim = ";"):
                return delim.join([mname, tname, ename, lname, value, error, epoch, speed, delta])
