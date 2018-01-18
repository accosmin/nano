import os
import re
import urllib3
import argparse
import subprocess

# todo: remove these and make it self-contained
import utils
import plotter

class config:
        """ utility to manage the configuration needed for an experiment:
                - paths (e.g. datasets, results, utilities)
                - available ML objects (e.g. solvers, loss functions)
        """
        def __init__(self, build_folder = "../build-release"):
                homedir = os.path.expanduser('~')
                self.expdir = os.path.join(homedir, "experiments", "results")
                self.dbdir = os.path.join(homedir, "experiments", "databases")

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_info = os.path.join(crtpath, build_folder, "apps", "info")
                self.app_train = os.path.join(crtpath, build_folder, "apps", "train")
                self.app_stats = os.path.join(crtpath, build_folder, "apps", "stats")
                self.app_builder = os.path.join(crtpath, build_folder, "apps", "builder")
                self.app_tabulate = os.path.join(crtpath, build_folder, "apps", "tabulate")

        def losses(self):
                """ available loss functions"""
                return ["cauchy",               # regression (robust to noise)
                        "square",               # regression
                        "classnll",             # classification (single label)
                        "s-logistic",           # classification (single label)
                        "m-logistic",           # classification (multi label)
                        "s-exponential",        # classification (single label)
                        "m-exponential"]        # classification (multi label)

        def enhancers(self):
                """ available data enhancing methods """
                return ["default",
                        "noise",
                        "warp",
                        "noclass"]

        def activations(self):
                """ available activation functions """
                return ["act-unit",
                        "act-sin",      # [-1, +1]
                        "act-tanh",     # [-1, +1]
                        "act-splus",    # [ 0,  1]
                        "act-snorm",    # [-1, +1]
                        "act-ssign",    # [-1, +1]
                        "act-sigm",     # [ 0,  1]
                        "act-pwave"]    # [-1, +1]

        def model(self, model_type, conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type):
                """ create a model using the built-in command line utility apps/builder """
                return  "--{} --act-type {} --conv3d-param {} --affine-param {} "\
                        "--imaps {} --irows {} --icols {} --omaps {} --orows {} --ocols {} ".format(
                        model_type, act_type, ','.join(map(str, conv3d_param)), ','.join(map(str, affine_param)),
                        imaps, irows, icols, omaps, orows, ocols)

        def mlp(self, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type="act-snorm"):
                """ create a MLP (multi-layer perceptron) """
                return self.model("mlp", [], affine_param, imaps, irows, icols, omaps, orows, ocols, act_type)

        def cnn(self, conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type="act-snorm"):
                """ create a CNN (convolution neural network) """
                return self.model("cnn", conv3d_param, affine_param, imaps, irows, icols, omaps, orows, ocols, act_type)

        def stoch_solvers(self):
                """ available stochastic solvers """
                return ["ag", "agfr", "aggr",
                        "sg", "sgm", "ngd", "svrg", "asgd",
                        "adagrad", "adadelta", "adam", "adaratio", "rmsprop"]

        def batch_solvers(self):
                """ available batch (line-search) solvers """
                return ["gd", "cgd", "lbfgs"]

        def stoch_trainer(self, solver, epochs = 100, patience = 32, epsilon = 1e-6, batch = 32, tune_epochs = 8):
                """ create a stochastic trainer """
                assert(solver in self.stoch_solvers())
                return {"trainer": "stoch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon,
                        "batch": batch, "tune_epochs": tune_epochs}

        def batch_trainer(self, solver, epochs = 100, patience = 32, epsilon = 1e-6):
                """ create a batch trainer """
                assert(solver in self.batch_solvers())
                return {"trainer": "batch", "solver": solver, "epochs": epochs, "patience": patience, "epsilon": epsilon}

        def loss(self, loss):
                """ create a loss """
                assert(loss in self.losses())
                return {"loss": loss}

        def enhancer(self, enhancer):
                """ create a data enhancer """
                assert(enhancer in self.enhancers())
                return {"enhancer": enhancer}

        def task(self, name):
                """ create a task """
                return {"task": name, "dir": os.path.join(self.dbdir, name)}

        def task_mnist(self):
                return self.task("mnist")

        def task_cifar10(self):
                return self.task("cifar10")

        def task_iris(self):
                return self.task("iris")

        def task_wine(self):
                return self.task("wine")

        def task_synth_charset(self, ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000):
                return {"task": "synth-charset", "type": ctype, "color": color, "irows": irows, "icols": icols, "count": count}

        def task_synth_nparity(self, n = 32, count = 10000):
                return {"task": "synth-nparity", "n": n, "count": count}

        def task_synth_affine_regression(self, isize = 32, osize = 32, noise = 0.0, count = 10000):
                return {"task": "synth-affine", "isize": isize, "osize": osize, "noise": noise, "count": count,
                        "type": "regression" }

        def task_synth_affine_classification(self, isize = 32, osize = 32, noise = 0.0, count = 10000):
                return {"task": "synth-affine", "isize": isize, "osize": osize, "noise": noise, "count": count,
                        "type": "classification" }

class downloader():
        """ utility to manage downloading well-known datasets (aka tasks)
        """
        def __init__(self):
                pass

        def get(self, url, dbdir):
                """ download the file at the given url to the given directory (by keeping its filename) """
                http = urllib3.PoolManager()
                r = http.request('GET', url, preload_content = False)
                path = dbdir + url.split('/')[-1]
                print("downloading", url, "to", path)
                chunk_size = 16 * 1024
                with open(path, 'wb') as out:
                        while True:
                                data = r.read(chunk_size)
                                if not data:
                                        break
                                out.write(data)
                r.release_conn()

        def mkdir(self, dbname):
                """ constructs and creates the directory to store a particular dataset """
                cfg = config()
                dbdir = cfg.dbdir + "/" + dbname + "/"
                os.makedirs(dbdir, exist_ok = True)
                return dbdir

        def get_iris(self):
                dbdir = self.mkdir("iris")
                self.get("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", dbdir)

        def get_wine(self):
                dbdir = self.mkdir("wine")
                self.get("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", dbdir)

        def get_svhn(self):
                dbdir = self.mkdir("svhn")
                self.get("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", dbdir)
                self.get("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", dbdir)
                self.get("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", dbdir)

        def get_mnist(self):
                dbdir = self.mkdir("mnist")
                self.get("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dbdir)
                self.get("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dbdir)
                self.get("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dbdir)
                self.get("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dbdir)

        def get_cifar10(self):
                dbdir = self.mkdir("cifar10")
                self.get("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", dbdir)

        def get_cifar100(self):
                dbdir = self.mkdir("cifar100")
                self.get("http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz", dbdir)

        def get_fashion_mnist(self):
                dbdir = self.mkdir("fashion-mnist")
                self.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", dbdir)
                self.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", dbdir)
                self.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", dbdir)
                self.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", dbdir)

class experiment:
        """ utility to manage an experiment consisting of a task and various models, trainers, enhancers and losses
                specified using json configurations
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

        def makedirs(self):
                """ create output directories """
                os.makedirs(self.dir, exist_ok = True)
                os.makedirs(self.dir_config, exist_ok = True)
                for trial in range(self.trials):
                        os.makedirs(self.dir + "/trial{}/".format(trial), exist_ok = True)

        def set_task(self, parameters):
                """ register the task (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "task.json")
                utils.save_json(json_path, parameters)
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
                utils.save_json(json_path, parameters)
                self.losses.append([name, json_path])

        def add_trainer(self, name, parameters):
                """ register a new trainer (.json config created from serializing parameters) """
                json_path = os.path.join(self.dir_config, "trainer_" + name + ".json")
                utils.save_json(json_path, parameters)
                self.trainers.append([name, json_path])

        def add_enhancer(self, name, parameters):
                """ register a new task enhancer (.json config created from serializing parameters) """
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

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="command line wrapper over the experimentation scripts")
        parser.add_argument("--download-tasks", action="store", help="regex of the task names to download", type=str)

        args = parser.parse_args()

        if args.download_tasks:
                getter = downloader()
                if re.match(args.download_tasks, "iris"): getter.get_iris()
                if re.match(args.download_tasks, "wine"): getter.get_wine()
                if re.match(args.download_tasks, "svhn"): getter.get_svhn()
                if re.match(args.download_tasks, "mnist"): getter.get_mnist()
                if re.match(args.download_tasks, "cifar10"): getter.get_cifar10()
                if re.match(args.download_tasks, "cifar100"): getter.get_cifar100()
                if re.match(args.download_tasks, "fashion-mnist"): getter.get_fashion_mnist()

