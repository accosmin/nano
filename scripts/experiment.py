import os
import re
import json
import time
import config
import plotter
import subprocess

class experiment:
        """ utility to manage an experiment consisting of a task and
                various models, trainers, enhancers and loss functions configured using JSON
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
                self.enhancers = []
                self.makedirs()

        def reg2str(self, reg):
                """ construct a string representation from a regular expression """
                return reg.replace(".*", "all").replace("*", "").replace(".", "") if reg else "all"

        def log(self, *messages):
                """ log messages in a nice format """
                print(time.strftime("[%Y-%m-%d %H:%M:%S]"), ' '.join(messages))

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

        def add_enhancer(self, name, parameters):
                """ register a new task enhancer (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "enhancer_" + name + ".json")
                self.save_json(json_path, parameters)
                self.enhancers.append([name, json_path])

        def path(self, dirname, mname, tname, ename, lname, extension, trial=None):
                basename = "M" + mname if mname else ""
                basename += "_T" + tname if tname else ""
                basename += "_E" + ename if ename else ""
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

        def filter_names(self, mname_reg, tname_reg, ename_reg, lname_reg):
                mnames = self.get_names(self.models, mname_reg)
                tnames = self.get_names(self.trainers, tname_reg)
                enames = self.get_names(self.enhancers, ename_reg)
                lnames = self.get_names(self.losses, lname_reg)
                return mnames, tnames, enames, lnames

        def filter_paths(self, dirname, mname_reg, tname_reg, ename_reg, lname_reg, extension, trial=None):
                paths = []
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, tname_reg, ename_reg, lname_reg)
                for mname, tname, ename, lname in [(u, x, y, z) for u in mnames for x in tnames for y in enames for z in lnames]:
                        paths.append(self.path(dirname, mname, tname, ename, lname, extension, trial))
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

        def train_one(self, mname, mparam, tname, tparam, ename, eparam, lname, lparam):
                # train the given configuration for multiple trials
                mpath = self.path(self.dir_models, mname, tname, ename, lname, "")
                cpath = self.path(self.dir_models, mname, tname, ename, lname, ".csv")
                lpath = self.path(self.dir_models, mname, tname, ename, lname, ".log")

                param = "\n --task " + self.task + "\n"
                param += " --loss " + lparam + "\n"
                param += " --model " + mparam + "\n"
                param += " --trainer " + tparam + "\n"
                param += " --enhancer " + eparam + "\n"
                param += " --basepath " + mpath + "\n"
                param += " --trials " + str(self.trials)

                with open(lpath, "w") as lfile:
                        self.log("running <", param, ">...")
                        subprocess.check_call((self.cfg.app_train + " " + param).split(), stdout = lfile)
                self.log("|--->training done, see <", lpath, ">")

                # summarize this configuration
                #subprocess.check_call((self.cfg.app_tabulate + " -i " +  cpath + " -p 3 -d ';' --stats").split())

                # plot the training history for each trial
                for trial in range(self.trials):
                        cpath = self.path(self.dir_models, mname, tname, ename, lname, ".csv", trial)
                        ppath = self.path(self.dir_models, mname, tname, ename, lname, ".pdf", trial)
                        self.plot_trial(cpath, ppath)

                # plot the training history for all trials on the same plot
                cpaths = []
                for trial in range(self.trials):
                        cpath = self.path(self.dir_models, mname, tname, ename, lname, ".csv", trial)
                        cpaths.append(cpath)
                ppath = self.path(self.dir_models, mname, tname, ename, lname, ".pdf")
                self.plot_trials(cpaths, ppath)

        def train_all(self):
                # run all possible configurations
                mdatas = self.models
                tdatas = self.trainers
                edatas = self.enhancers
                ldatas = self.losses
                for mdata, tdata, edata, ldata in [(u, x, y, z) for u in mdatas for x in tdatas for y in edatas for z in ldatas]:
                        self.train_one(
                                self.get_name(mdata), self.get_config(mdata),
                                self.get_name(tdata), self.get_config(tdata),
                                self.get_name(edata), self.get_config(edata),
                                self.get_name(ldata), self.get_config(ldata))
                        self.log()

        def summarize(self, mname, mname_reg, tname, tname_reg, ename, ename_reg, lname, lname_reg, names):
                # compare configurations for each trial: plot the training state evaluation
                for trial in range(self.trials):
                        paths = self.filter_paths(self.dir_models, mname_reg, tname_reg, ename_reg, lname_reg, ".csv", trial)
                        ppath = self.path(self.dir, mname, tname, ename, lname, ".pdf", trial)
                        self.plot_trials(paths, ppath)

                # compare configurations across all trials: boxplot the results
                paths = self.filter_paths(self.dir_models, mname_reg, tname_reg, ename_reg, lname_reg, ".csv")
                ppath = self.path(self.dir, mname, tname, ename, lname, ".pdf")
                self.plot_configs(paths, ppath, names)

        def summarize_by_models(self, mname_reg = ".*"):
                mname = self.reg2str(mname_reg)
                mnames, tnames, enames, lnames = self.filter_names(mname_reg, None, None, None)
                for tname, ename, lname in [(x, y, z) for x in tnames for y in enames for z in lnames]:
                        self.summarize(mname, mname_reg, tname, tname, ename, ename, lname, lname, mnames)

        def summarize_by_trainers(self, tname_reg = ".*"):
                tname = self.reg2str(tname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, tname_reg, None, None)
                for mname, ename, lname in [(x, y, z) for x in mnames for y in enames for z in lnames]:
                        self.summarize(mname, mname, tname, tname_reg, ename, ename, lname, lname, tnames)

        def summarize_by_enhancers(self, ename_reg = ".*"):
                ename = self.reg2str(ename_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, ename_reg, None)
                for mname, tname, lname in [(x, y, z) for x in mnames for y in tnames for z in lnames]:
                        self.summarize(mname, mname, tname, tname, ename, ename_reg, lname, lname, enames)

        def summarize_by_losses(self, lname_reg = ".*"):
                lname = self.reg2str(lname_reg)
                mnames, tnames, enames, lnames = self.filter_names(None, None, None, lname_reg)
                for mname, tname, ename in [(x, y, z) for x in mnames for y in tnames for z in enames]:
                        self.summarize(mname, mname, tname, tname, ename, ename, lname, lname_reg, lnames)
