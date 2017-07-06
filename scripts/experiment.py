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
        def __init__(self, task, outdir, trials = 10):
                self.cfg = config.config()
                self.task = task
                self.dir = outdir
                self.models = []
                self.trainers = []
                self.iterators = []
                self.losses = []
                self.trials = trials

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
                basepath += "_M" + mname if mname else ""
                basepath += "_T" + tname if tname else ""
                basepath += "_I" + iname if iname else ""
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

        def filter_names(self, mname_reg, tname_reg, iname_reg, lname_reg):
                mnames = self.names(self.models, mname_reg)
                tnames = self.names(self.trainers, tname_reg)
                inames = self.names(self.iterators, iname_reg)
                lnames = self.names(self.losses, lname_reg)
                return mnames, tnames, inames, lnames

        def filter_paths(self, trial, mname_reg, tname_reg, iname_reg, lname_reg, extension):
                paths = []
                mnames, tnames, inames, lnames = self.filter_names(mname_reg, tname_reg, iname_reg, lname_reg)
                for mname, tname, iname, lname in [(u, x, y, z) for u in mnames for x in tnames for y in inames for z in lnames]:
                        paths.append(self.path(trial, mname, tname, iname, lname, extension))
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

        def train_trials(self, mname, mparam, tname, tparam, iname, iparam, lname, lparam):
                # train the given configuration for multiple trials
                for trial in range(self.trials):
                        os.makedirs(self.dir, exist_ok = True)
                        mpath = self.path(trial, mname, tname, iname, lname, ".model")
                        spath = self.path(trial, mname, tname, iname, lname, ".state")
                        lpath = self.path(trial, mname, tname, iname, lname, ".log")
                        ppath = self.path(trial, mname, tname, iname, lname, ".pdf")

                        param = self.task + " " + mparam + " " + tparam + " " + iparam + " " + lparam + " --model-file " + mpath
                        self.train_one(param, lpath)
                        self.plot_one(spath, ppath)
                        self.log()

                # export the results from multiple trials as csv
                cpath = self.path(None, mname, tname, iname, lname, ".csv")
                spath = self.path(None, mname, tname, iname, lname, ".stats")
                with open(cpath, "w") as cfile, open(spath, "w") as sfile:
                        print(self.get_csv_header(), file = cfile)
                        print(self.get_csv_header(), file = sfile)
                        values, errors, epochs, speeds, deltas = self.get_logs( mname, tname, iname, lname)
                        for value, error, epoch, speed, delta in zip(values, errors, epochs, speeds, deltas):
                                print(self.get_csv_row(mname, tname, iname, lname, value, error, epoch, speed, delta), file = cfile)
                        value_stats, error_stats, epoch_stats, speed_stats, delta_stats = self.get_log_stats(values, errors, epochs, speeds, deltas)
                        print(self.get_csv_row(mname, tname, iname, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats), file = sfile)

                # display basic statistics for the results from multiple trials
                lpath = self.path(None, mname, tname, iname, lname, ".log")
                self.tabulate(spath, lpath)

        def train_all(self):
                # run all possible configurations
                mdatas = self.models
                tdatas = self.trainers
                idatas = self.iterators
                ldatas = self.losses
                for mdata, tdata, idata, ldata in [(u, x, y, z) for u in mdatas for x in tdatas for y in idatas for z in ldatas]:
                        self.train_trials(
                                self.name(mdata), self.config(mdata),
                                self.name(tdata), self.config(tdata),
                                self.name(idata), self.config(idata),
                                self.name(ldata), self.config(ldata))

        def summarize_trials(self, mname, mname_reg, tname, tname_reg, iname, iname_reg, lname, lname_reg, names):
                # compare configurations for each trial: plot the training state evaluation
                for trial in range(self.trials):
                        self.plot_many(
                                self.filter_paths(trial, mname_reg, tname_reg, iname_reg, lname_reg, ".state"),
                                self.path(trial, mname, tname, iname, lname, ".pdf"))

                # compare configurations across all trials: boxplot the results
                ppath = self.path(None, mname, tname, iname, lname, ".pdf")

                self.plot_trial(
                        self.filter_paths(None, mname_reg, tname_reg, iname_reg, lname_reg, ".csv"),
                        ppath, names)

                # compare configurations across all trials: display basic statistics
                spath = self.path(None, mname, tname, iname, lname, ".stats")
                with open(spath, "w") as sfile:
                        print(self.get_csv_header(), file = sfile)
                        mnames, tnames, inames, lnames = self.filter_names(mname_reg, tname_reg, iname_reg, lname_reg)
                        for mname, tname, iname, lname in [(u, x, y, z) for u in mnames for x in tnames for y in inames for z in lnames]:
                                values, errors, epochs, speeds, deltas = self.get_logs(mname, tname, iname, lname)
                                value_stats, error_stats, epoch_stats, speed_stats, delta_stats = self.get_log_stats(values, errors, epochs, speeds, deltas)
                                print(self.get_csv_row(mname, tname, iname, lname, value_stats, error_stats, epoch_stats, speed_stats, delta_stats), file = sfile)

                lpath = self.path(None, mname, tname, iname, lname, ".log")
                self.tabulate(spath, lpath)

        def summarize_by_models(self, mname_reg = ".*"):
                mname = self.reg2str(mname_reg)
                mnames, tnames, inames, lnames = self.filter_names(mname_reg, None, None, None)
                for tname, iname, lname in [(x, y, z) for x in tnames for y in inames for z in lnames]:
                        self.summarize_trials(mname, mname_reg, tname, tname, iname, iname, lname, lname, mnames)

        def summarize_by_trainers(self, tname_reg = ".*"):
                tname = self.reg2str(tname_reg)
                mnames, tnames, inames, lnames = self.filter_names(None, tname_reg, None, None)
                for mname, iname, lname in [(x, y, z) for x in mnames for y in inames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname_reg, iname, iname, lname, lname, tnames)

        def summarize_by_iterators(self, iname_reg = ".*"):
                iname = self.reg2str(iname_reg)
                mnames, tnames, inames, lnames = self.filter_names(None, None, iname_reg, None)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname, iname, iname_reg, lname, lname, inames)

        def summarize_by_losses(self, lname_reg = ".*"):
                lname = self.reg2str(lname_reg)
                mnames, tnames, inames, lnames = self.filter_names(None, None, None, lname_reg)
                for mname, tname, iname in [(x, y, z) for x in mnames for y in tnames for z in inames]:
                        self.summarize_trials(mname, mname, tname, tname, iname, iname, lname, lname_reg, lnames)

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

        def get_logs(self, mname, tname, iname, lname):
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(self.trials):
                        lpath = self.path(trial, mname, tname, iname, lname, ".log")
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
                return delim.join(["model", "trainer", "iterator", "loss", "test value", "test error", "epochs", "convergence speed", "duration (sec)"])

        def get_csv_row(self, mname, tname, iname, lname, value, error, epoch, speed, delta, delim = ";"):
                return delim.join([mname, tname, iname, lname, value, error, epoch, speed, delta])
