import os
import re
import json
import config
import plotter
import logging
import subprocess

class experiment:
        """ utility to manage an experiment consisting of a task and
                various models, trainers and loss functions configured using JSON
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
                self.trainers = []
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
                parameters += " --json {}".format(json_path)
                subprocess.check_call((self.cfg.app_builder + " " + parameters).split(), stdout=subprocess.DEVNULL)
                self.models.append([name, json_path])

        def add_loss(self, name, parameters):
                """ register a new loss function (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "loss_" + name + ".json")
                self.save_json(json_path, parameters)
                self.losses.append([name, json_path])

        def add_trainer(self, name, parameters):
                """ register a new trainer (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "trainer_" + name + ".json")
                self.save_json(json_path, parameters)
                self.trainers.append([name, json_path])

        def path(self, dirname, mname, tname, lname, extension, trial=None):
                basename = "M" + mname if mname else ""
                basename += "_T" + tname if tname else ""
                basename += "_L" + lname if lname else ""
                basename += "" if trial is None else "_trial{}".format(trial + 1)
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

        def filter_names(self, mname_reg, tname_reg, lname_reg):
                mnames = self.get_names(self.models, mname_reg)
                tnames = self.get_names(self.trainers, tname_reg)
                lnames = self.get_names(self.losses, lname_reg)
                return mnames, tnames, lnames

        def filter_paths(self, dirname, mname_reg, tname_reg, lname_reg, extension, trial=None):
                paths = []
                mnames, tnames, lnames = self.filter_names(mname_reg, tname_reg, lname_reg)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        paths.append(self.path(dirname, mname, tname, lname, extension, trial))
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
                self.log("{}{}{}{}".format("-" * 42, "-" * 24, "-" * 36, "-" * 36))
                self.log("{}{}{}{}".format("".ljust(42), "test error".rjust(24), "epochs".rjust(36), "seconds".rjust(36)))
                self.log("{}{}{}{}".format("-" * 42, "-" * 24, "-" * 36, "-" * 36))
                for cpath in cpaths:
                        name = os.path.basename(cpath).replace(".csv", "").ljust(42)
                        basecmd = self.cfg.app_tabulate + " -i " + cpath + " -p 3 -d ';' --stats "
                        error_stats = subprocess.check_output(basecmd + " -c 7", shell=True).decode('ascii').rstrip().rjust(24)
                        epoch_stats = subprocess.check_output(basecmd + " -c 1", shell=True).decode('ascii').rstrip().rjust(36)
                        time_stats = subprocess.check_output(basecmd + " -c 10", shell=True).decode('ascii').rstrip().rjust(36)
                        self.log("{}{}{}{}".format(name, error_stats, epoch_stats, time_stats))
                self.log("{}{}{}{}".format("-" * 42, "-" * 24, "-" * 36, "-" * 36))

        def train_one(self, mname, mparam, tname, tparam, lname, lparam):
                # train the given configuration for multiple trials
                mpath = self.path(self.dir_models, mname, tname, lname, "")
                cpath = self.path(self.dir_models, mname, tname, lname, ".csv")
                lpath = self.path(self.dir_models, mname, tname, lname, ".log")

                param = "\n --task " + self.task + "\n"
                param += " --loss " + lparam + "\n"
                param += " --model " + mparam + "\n"
                param += " --trainer " + tparam + "\n"
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
                        cpath = self.path(self.dir_models, mname, tname, lname, ".csv", trial)
                        ppath = self.path(self.dir_models, mname, tname, lname, ".pdf", trial)
                        self.plot_trial(cpath, ppath)

                # plot the training history for all trials on the same plot
                cpaths = []
                for trial in range(self.trials):
                        cpath = self.path(self.dir_models, mname, tname, lname, ".csv", trial)
                        cpaths.append(cpath)
                ppath = self.path(self.dir_models, mname, tname, lname, ".pdf")
                self.plot_trials(cpaths, ppath)
                self.log()

        def train_all(self):
                # run all possible configurations
                mdatas = self.models
                tdatas = self.trainers
                ldatas = self.losses
                for mdata, tdata, ldata in [(x, y, z) for x in mdatas for y in tdatas for z in ldatas]:
                        self.train_one(
                                self.get_name(mdata), self.get_config(mdata),
                                self.get_name(tdata), self.get_config(tdata),
                                self.get_name(ldata), self.get_config(ldata))

        def summarize(self, mname, mname_reg, tname, tname_reg, lname, lname_reg, names):
                # sumarize these configurations
                paths = self.filter_paths(self.dir_models, mname_reg, tname_reg, lname_reg, ".csv")
                self.print_stats(paths)

                # compare configurations for each trial: plot the training state evaluation
                for trial in range(self.trials):
                        paths = self.filter_paths(self.dir_models, mname_reg, tname_reg, lname_reg, ".csv", trial)
                        ppath = self.path(self.dir, mname, tname, lname, ".pdf", trial)
                        self.plot_trials(paths, ppath)

                # compare configurations across all trials: boxplot the results
                paths = self.filter_paths(self.dir_models, mname_reg, tname_reg, lname_reg, ".csv")
                ppath = self.path(self.dir, mname, tname, lname, ".pdf")
                self.plot_configs(paths, ppath, names)
                self.log()

        def summarize_by_models(self, mname, mname_reg = ".*"):
                mnames, tnames, lnames = self.filter_names(mname_reg, None, None)
                for tname, lname in [(x, y) for x in tnames for y in lnames]:
                        self.summarize(mname, mname_reg, tname, tname, lname, lname, mnames)

        def summarize_by_trainers(self, tname, tname_reg = ".*"):
                mnames, tnames, lnames = self.filter_names(None, tname_reg, None)
                for mname, lname in [(x, y) for x in mnames for y in lnames]:
                        self.summarize(mname, mname, tname, tname_reg, lname, lname, tnames)

        def summarize_by_losses(self, lname, lname_reg = ".*"):
                mnames, tnames, lnames = self.filter_names(None, None, lname_reg)
                for mname, tname in [(x, y) for x in mnames for y in tnames]:
                        self.summarize(mname, mname, tname, tname, lname, lname_reg, lnames)
