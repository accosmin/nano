import os
import re
import utils
import config
import plotter
import subprocess
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class experiment:
        """ an experiment run/trial consists of:
                - a trial (fold index or None if cumulating results from multiple trials)
                - a model (name + command line parameters to apps/builder)
                - a trainer (name + json parameters)
                - an enhancer (name + json parameters)
                - a loss function (name)
        """
        def __init__(self, outdir, trials = 10):
                self.cfg = config.config()
                self.dir = outdir
                self.dir_config = outdir + "/config"
                self.trials = trials
                self.runs = []
                self.losses = []
                self.models = []
                self.trainers = []
                self.enhancers = []
                self.makedirs()

        """ create output directories """
        def makedirs(self):
                os.makedirs(self.dir, exist_ok = True)
                os.makedirs(self.dir_config, exist_ok = True)
                for trial in range(self.trials):
                        os.makedirs(self.dir + "/trial{}/".format(trial), exist_ok = True)

        """ register the task (.json config created from serializing parameters) """
        def set_task(self, parameters):
                json_path = os.path.join(self.dir_config, "task.json")
                utils.save_json(json_path, parameters)
                self.task = json_path

        """ register a new model (.json config created using apps/builder) """
        def add_model(self, name, parameters):
                json_path = os.path.join(self.dir_config, "model_" + name + ".json")
                parameters += " --json {}".format(json_path)
                subprocess.check_call((self.cfg.app_builder + " " + parameters).split(), stdout=subprocess.DEVNULL)
                self.models.append([name, json_path])

        """ register a new loss function (.json config created from serializing parameters) """
        def add_loss(self, name, parameters):
                json_path = os.path.join(self.dir_config, "loss_" + name + ".json")
                utils.save_json(json_path, parameters)
                self.losses.append([name, json_path])

        """ register a new trainer (.json config created from serializing parameters) """
        def add_trainer(self, name, parameters):
                json_path = os.path.join(self.dir_config, "trainer_" + name + ".json")
                utils.save_json(json_path, parameters)
                self.trainers.append([name, json_path])

        """ register a new task enhancer (.json config created from serializing parameters) """
        def add_enhancer(self, name, parameters):
                json_path = os.path.join(self.dir_config, "enhancer_" + name + ".json")
                utils.save_json(json_path, parameters)
                self.enhancers.append([name, json_path])

        def path(self, trial, mname, tname, ename, lname, extension):
                basepath = self.dir
                if not (trial is None):
                        basepath += "/trial{}/".format(trial)
                else:
                        basepath += "/"
                basepath += "M" + mname if mname else ""
                basepath += "_T" + tname if tname else ""
                basepath += "_E" + ename if ename else ""
                basepath += "_L" + lname if lname else ""
                basepath += extension
                return basepath

        def get_name(self, name_config):
                return name_config[0]

        def get_config(self, name_config):
                return name_config[1]

        def get_names(self, names_configs, name_reg = None):
                names = []
                for name_config in names_configs:
                        name = self.get_name(name_config)
                        if name_reg == None or re.match(name_reg, name):
                                names.append(name)
                return names

        def filter_names(self, mname_reg, tname_reg, ename_reg, lname_reg):
                mnames = self.get_names(self.models, mname_reg)
                tnames = self.get_names(self.trainers, tname_reg)
                enames = self.get_names(self.enhancers, ename_reg)
                lnames = self.get_names(self.losses, lname_reg)
                return mnames, tnames, enames, lnames

        def filter_paths(self, trial, mname_reg, tname_reg, ename_reg, lname_reg, extension):
                paths = []
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, tname_reg, ename_reg, lname_reg)
                for mname, tname, ename, lname in [(u, x, y, z) for u in mnames for x in tnames for y in enames for z in lnames]:
                        paths.append(self.path(trial, mname, tname, ename, lname, extension))
                return paths

        def train_one(self, param, lpath):
                with open(lpath, "w") as lfile:
                        utils.log("running <", param, ">...")
                        subprocess.check_call((self.cfg.app_train + " " + param).split(), stdout = lfile)
                utils.log("|--->training done, see <", lpath, ">")

        def plot_one(self, spath, ppath):
                plotter.plot_state_one(spath, ppath)
                utils.log("|--->plotting done, see <", ppath, ">")

        def plot_many(self, spaths, ppath):
                plotter.plot_state_many(spaths, ppath)
                utils.log("|--->plotting done, see <", ppath, ">")

        def plot_trial(self, spaths, ppath, names):
                plotter.plot_trial_many(spaths, ppath, names)
                utils.log("|--->plotting done, see <", ppath, ">")

        def tabulate(self, cpath, lpath):
                with open(lpath, "w") as lfile:
                        subprocess.check_call((self.cfg.app_tabulate + " -i " + cpath + " -d \";\"").split(), stdout = lfile)

                with open(lpath, "r") as lfile:
                        utils.log()
                        print(lfile.read())

        def summarize_by_models(self, mname_reg = ".*"):
                mname = utils.reg2str(mname_reg)

        def train_trials(self, mname, mparam, tname, tparam, ename, eparam, lname, lparam):
                # train the given configuration for multiple trials
                for trial in range(self.trials):
                        mpath = self.path(trial, mname, tname, ename, lname, "")
                        spath = self.path(trial, mname, tname, ename, lname, ".state")
                        lpath = self.path(trial, mname, tname, ename, lname, ".log")
                        ppath = self.path(trial, mname, tname, ename, lname, ".pdf")

                        param = "  --task " + self.task
                        param += " --model " + mparam
                        param += " --trainer " + tparam
                        param += " --enhancer " + eparam
                        param += " --loss " + lparam
                        param += " --basepath " + mpath

                        self.train_one(param, lpath)
                        self.plot_one(spath, ppath)
                        utils.log()

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
                                self.get_name(mdata), self.get_config(mdata),
                                self.get_name(tdata), self.get_config(tdata),
                                self.get_name(edata), self.get_config(edata),
                                self.get_name(ldata), self.get_config(ldata))

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
                mname = utils.reg2str(mname_reg)
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, None, None, None)
                for tname, ename, lname in [(x, y, z) for x in tnames for y in enames for z in lnames]:
                        self.summarize_trials(mname, mname_reg, tname, tname, ename, ename, lname, lname, mnames)

        def summarize_by_trainers(self, tname_reg = ".*"):
                tname = utils.reg2str(tname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, tname_reg, None, None)
                for mname, ename, lname in [(x, y, z) for x in mnames for y in enames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname_reg, ename, ename, lname, lname, tnames)

        def summarize_by_enhancers(self, ename_reg = ".*"):
                ename = utils.reg2str(ename_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, ename_reg, None)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        self.summarize_trials(mname, mname, tname, tname, ename, ename_reg, lname, lname, enames)

        def summarize_by_losses(self, lname_reg = ".*"):
                lname = utils.reg2str(lname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, None, lname_reg)
                for mname, tname, ename in [(x, y, z) for x in mnames for y in tnames for z in enames]:
                        self.summarize_trials(mname, mname, tname, tname, ename, ename, lname, lname_reg, lnames)

        def get_logs(self, mname, tname, ename, lname):
                values = []
                errors = []
                epochs = []
                speeds = []
                deltas = []
                for trial in range(self.trials):
                        lpath = self.path(trial, mname, tname, ename, lname, ".log")
                        value, error, epoch, speed, delta = utils.get_log(lpath)
                        values.append(value)
                        errors.append(error)
                        epochs.append(epoch)
                        speeds.append(speed)
                        deltas.append(delta)
                return values, errors, epochs, speeds, deltas

        def get_log_stats(self, values, errors, epochs, speeds, deltas):
                cmdline = self.cfg.app_stats + " -p 3"
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
