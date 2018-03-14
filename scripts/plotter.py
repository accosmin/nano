import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_trial_csv(path, delimiter = ";"):
        """ load csv file for a configuration trial with the following format:
        (epoch, [train|valid|test] x [loss|error], xnorm, gnorm, seconds)+
        """
        name = os.path.basename(path).replace(".csv", "")
        data = mlab.csv2rec(path, delimiter=delimiter, names=None)
        return name, data

def load_config_csv(path, delimiter = ";"):
        """ load csv file for a configuration summary with the following format:
        (trial, optimum epoch, [train|valid|test] x [loss|error], xnorm, gnorm, seconds, speed)+
        """
        name = os.path.basename(path).replace(".csv", "")
        data = mlab.csv2rec(path, delimiter=delimiter, names=None)
        return name, data

def load_csvs(paths, loader, delimiter = ";"):
        names, datas = [], []
        for path in paths:
                name, data = loader(path, delimiter)
                names.append(name)
                datas.append(data)
        return names, datas

def plot_trial(spath, ppath):
        """ plot the training evolution of a model """
        title, data = load_trial_csv(spath)
        with PdfPages(ppath) as pdf:
                # (train, validation, test) loss value and error
                for ynames in (["train_loss", "valid_loss", "test_loss"], ["train_error", "valid_error", "test_error"]):
                        xname, yname0, yname1, yname2 = "epoch", ynames[0], ynames[1], ynames[2]
                        plt.plot(data[xname], data[yname0], "r-", label = yname0)
                        plt.plot(data[xname], data[yname1], "g-", label = yname1)
                        plt.plot(data[xname], data[yname2], "b-", label = yname2)
                        plt.title(title, weight = "bold")
                        plt.xlabel(xname, fontsize = "smaller")
                        plt.ylabel(yname0.replace("train_", ""), fontsize = "smaller")
                        plt.legend(fontsize = "smaller")
                        plt.grid(True, linestyle='--')
                        pdf.savefig()
                        plt.close()

                # xnorm and gnorm
                for yname in ("xnorm", "gnorm"):
                        xname = "epoch"
                        plt.plot(data[xname], data[yname], "k-", label = yname)
                        plt.title(title, weight = "bold")
                        plt.xlabel(xname, fontsize = "smaller")
                        plt.ylabel(yname, fontsize = "smaller")
                        plt.legend(fontsize = "smaller")
                        plt.grid(True, linestyle='--')
                        pdf.savefig()
                        plt.close()

def plot_trials(spaths, ppath):
        """ plot the training evolution of multiple models on the same plot """
        names, datas = load_csvs(spaths, load_trial_csv)
        with PdfPages(ppath) as pdf:
                for yname in ("train_loss", "train_error", "valid_loss", "valid_error", "test_loss", "test_error", "xnorm", "gnorm"):
                        for xname in ("epoch", "seconds"):
                                for data, name in zip(datas, names):
                                        plt.plot(data[xname], data[yname], label = name)
                                plt.xlabel(xname, fontsize = "smaller")
                                plt.ylabel(yname.replace("train_", "").replace("valid_", "").replace("test_", ""), fontsize = "smaller")
                                plt.title(yname, weight = "bold")
                                plt.legend(fontsize = "smaller")
                                plt.grid(True, linestyle='--')
                                pdf.savefig()
                                plt.close()

def plot_configs(spaths, ppath, names):
        """ plot the test results of multiple models on the sample plot """
        __, datas = load_csvs(spaths, load_config_csv)
        assert(len(names) == len(datas))
        title = os.path.basename(ppath).replace(".pdf", "").replace("result_", "")
        with PdfPages(ppath) as pdf:
                for yname in ("epoch", "train_loss", "train_error", "valid_loss", "valid_error", "test_loss", "test_error", "xnorm", "gnorm", "seconds", "speed"):
                        bdata = []
                        for data in datas:
                                bdata.append(data[yname])
                        plt.boxplot(bdata, labels = names)
                        plt.title(title, weight = "bold")
                        plt.ylabel(yname, fontsize = "smaller")
                        plt.grid(True, linestyle='--')
                        pdf.savefig()
                        plt.close()
