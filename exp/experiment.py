import os
import re
import json
import config
import plotter
import logging
import subprocess

class experiment:
        """ utility to manage an experiment consisting of a task and
                various models and loss functions configured using JSON
        """
        def __init__(self, outdir, trials = 10):
                self.cfg = config.config()
                self.dir = outdir
                self.dir_config = outdir + "/config"
                self.dir_models = outdir + "/models"
                self.trials = trials
                self.runs = []
                self.losses = []
                self.models = []
                self.makedirs()

                logging.basicConfig(
                        format="%(asctime)s %(message)s",
                        level=logging.DEBUG,
                        handlers=[
                                logging.FileHandler(os.path.join(outdir, "log")),
                                logging.StreamHandler()
                        ])

        def log(self, *messages):
                """ log messages in a nice format """
                logging.info(' '.join(messages))

        def save_json(self, path, parameters, indent=4):
                """ save the given parameters as json """
                with open(path, "w") as output:
                        output.write(json.dumps(parameters, indent=indent))

        def makedirs(self):
                """ create output directories """
                os.makedirs(self.dir, exist_ok = True)
                os.makedirs(self.dir_config, exist_ok = True)
                os.makedirs(self.dir_models, exist_ok = True)

        def set_task(self, parameters):
                """ register the task (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "task.json")
                self.save_json(json_path, parameters)
                self.task = json_path

        def add_model(self, name, parameters):
                """ register a new model (.json config created using apps/builder) """
                json_path = os.path.join(self.dir_config, "model_" + name + ".json")
                self.save_json(json_path, parameters)
                self.models.append([name, json_path])

        def add_loss(self, name, parameters):
                """ register a new loss function (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "loss_" + name + ".json")
                self.save_json(json_path, parameters)
                self.losses.append([name, json_path])

        def path(self, dirname, mname, lname, extension):
                basename = "M" + mname if mname else ""
                basename += "_L" + lname if lname else ""
                basename += extension
                return os.path.join(dirname, basename)

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

        def filter_names(self, mname_reg, lname_reg):
                mnames = self.get_names(self.models, mname_reg)
                lnames = self.get_names(self.losses, lname_reg)
                return mnames, lnames

        def filter_paths(self, dirname, mname_reg, lname_reg, extension):
                paths = []
                mnames, lnames = self.filter_names(mname_reg, lname_reg)
                for mname, lname in [(x, y) for x in mnames for y in lnames]:
                        paths.append(self.path(dirname, mname, lname, extension, trial))
                return paths

        def plot_trial(self, cpath, ppath):
                plotter.plot_trial(cpath, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_trials(self, cpaths, ppath):
                plotter.plot_trials(cpaths, ppath)
                self.log("|--->plotting done, see <", ppath, ">")

        def plot_configs(self, cpaths, ppath, names):
                plotter.plot_configs(cpaths, ppath, names)
                self.log("|--->plotting done, see <", ppath, ">")

        def print_stats(self, cpaths):
                self.log("{}{}{}{}".format("-" * 36, "-" * 24, "-" * 24, "-" * 24))
                self.log("{}{}{}{}".format("configuration".ljust(36), "test error".rjust(24), "epochs".rjust(24), "seconds".rjust(24)))
                self.log("{}{}{}{}".format("-" * 36, "-" * 24, "-" * 24, "-" * 24))
                for cpath in cpaths:
                        name = os.path.basename(cpath).replace(".csv", "").ljust(36)
                        basecmd = self.cfg.app_tabulate + " -i " + cpath + " -d ';' --stats "
                        error_stats = subprocess.check_output(basecmd + " -p 3 -c 7", shell=True).decode('ascii').rstrip().rjust(24)
                        epoch_stats = subprocess.check_output(basecmd + " -p 0 -c 1", shell=True).decode('ascii').rstrip().rjust(24)
                        time_stats = subprocess.check_output(basecmd + " -p 0 -c 10", shell=True).decode('ascii').rstrip().rjust(24)
                        self.log("{}{}{}{}".format(name, error_stats, epoch_stats, time_stats))
                self.log("{}{}{}{}".format("-" * 36, "-" * 24, "-" * 24, "-" * 24))

        def train_one(self, mname, mparam, lname, lparam):
                # train the given configuration for multiple trials
                mpath = self.path(self.dir_models, mname, lname, "")
                cpath = self.path(self.dir_models, mname, lname, ".csv")
                lpath = self.path(self.dir_models, mname, lname, ".log")

                param = "\n --task " + self.task + "\n"
                param += " --loss " + lparam + "\n"
                param += " --model " + mparam + "\n"
                param += " --basepath " + mpath + "\n"
                param += " --trials " + str(self.trials)

                with open(lpath, "w") as lfile:
                        self.log("running <", param, ">...")
                        subprocess.check_call((self.cfg.app_train + " " + param).split(), stdout = lfile)
                self.log("|--->training done, see <", lpath, ">")

                # summarize this configuration
                self.print_stats([cpath])

                # plot the training history for each trial
                for trial in range(self.trials):
                        cpath = self.path(self.dir_models, mname, lname, ".csv", trial)
                        ppath = self.path(self.dir_models, mname, lname, ".pdf", trial)
                        self.plot_trial(cpath, ppath)

                # plot the training history for all trials on the same plot
                cpaths = []
                for trial in range(self.trials):
                        cpath = self.path(self.dir_models, mname, lname, ".csv", trial)
                        cpaths.append(cpath)
                ppath = self.path(self.dir_models, mname, lname, ".pdf")
                self.plot_trials(cpaths, ppath)
                self.log()

        def train_all(self):
                # run all possible configurations
                mdatas = self.models
                ldatas = self.losses
                for mdata, tdata, ldata in [(x, y, z) for x in mdatas for y in tdatas for z in ldatas]:
                        self.train_one(
                                self.get_name(mdata), self.get_config(mdata),
                                self.get_name(tdata), self.get_config(tdata),
                                self.get_name(ldata), self.get_config(ldata))

        def summarize(self, mname, mname_reg, lname, lname_reg, names):
                # sumarize these configurations
                paths = self.filter_paths(self.dir_models, mname_reg, lname_reg, ".csv")
                self.print_stats(paths)

                # compare configurations for each trial: plot the training state evaluation
                for trial in range(self.trials):
                        paths = self.filter_paths(self.dir_models, mname_reg, lname_reg, ".csv", trial)
                        ppath = self.path(self.dir, mname, lname, ".pdf", trial)
                        self.plot_trials(paths, ppath)

                # compare configurations across all trials: boxplot the results
                paths = self.filter_paths(self.dir_models, mname_reg, lname_reg, ".csv")
                ppath = self.path(self.dir, mname, lname, ".pdf")
                self.plot_configs(paths, ppath, names)
                self.log()

        def summarize_by_models(self, mname, mname_reg = ".*"):
                mnames, lnames = self.filter_names(mname_reg, None)
                for lname in lnames:
                        self.summarize(mname, mname_reg, lname, lname, mnames)

        def summarize_by_losses(self, lname, lname_reg = ".*"):
                mnames, lnames = self.filter_names(None, lname_reg)
                for mname in mnames:
                        self.summarize(mname, mname, lname, lname_reg, lnames)
