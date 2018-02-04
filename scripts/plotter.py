import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
                for col in (0, 1):
                        # x axis - epoch/iteration index
                        xname = data.dtype.names[0]
                        xlabel = xname
                        # y axis - train/validation/test datasets
                        yname0 = data.dtype.names[col + 1]
                        yname1 = data.dtype.names[col + 3]
                        yname2 = data.dtype.names[col + 5]
                        ylabel = yname0.replace("train_", "")
                        # plot
                        plt.xlabel(xlabel, fontsize = "smaller")
                        plt.ylabel(ylabel, fontsize = "smaller")
                        plt.title(title, weight = "bold")
                        plt.plot(data[xname], data[yname0], "r-", label = yname0)
                        plt.plot(data[xname], data[yname1], "g-", label = yname1)
                        plt.plot(data[xname], data[yname2], "b-", label = yname2)
                        plt.legend(fontsize = "smaller")
                        plt.grid(True, linestyle='--')
                        pdf.savefig()
                        plt.close()
                # xnorm and gnorm
                for col in (6, 7):
                        # x axis - epoch/iteration index
                        xname = data.dtype.names[0]
                        xlabel = xname
                        # y axis
                        yname = data.dtype.names[col + 1]
                        ylabel = yname
                        # plot
                        plt.xlabel(xlabel, fontsize = "smaller")
                        plt.ylabel(ylabel, fontsize = "smaller")
                        plt.title(title, weight = "bold")
                        plt.plot(data[xname], data[yname], "k-", label = yname)
                        plt.legend(fontsize = "smaller")
                        plt.grid(True, linestyle='--')
                        pdf.savefig()
                        plt.close()

def plot_trials_wrt(names, datas, pdf, xcol, ycol):
        """ plot the training evolution of multiple models on the same plot """
        colnames = datas[0].dtype.names
        title = colnames[ycol]
        # x axis - epoch/iteration index
        xname = colnames[xcol]
        xlabel = xname
        # y axis - train/validation/test datasets
        yname = colnames[ycol]
        ylabel = yname.replace("train_", "").replace("valid_", "").replace("test_", "")
        # plot
        plt.xlabel(xlabel, fontsize = "smaller")
        plt.ylabel(ylabel, fontsize = "smaller")
        plt.title(title, weight = "bold")
        for data, name in zip(datas, names):
                plt.plot(data[xname], data[yname], label = name)
        plt.legend(fontsize = "smaller")
        plt.grid(True, linestyle='--')
        pdf.savefig()
        plt.close()

def plot_trials(spaths, ppath):
        names, datas = load_csvs(spaths, load_trial_csv)
        with PdfPages(ppath) as pdf:
                for col in (1, 2, 3, 4, 5, 6, 8, 9):
                        # plot wrt epoch/iteration number
                        plot_trials_wrt(names, datas, pdf, 0, col)
                        # plot wrt time
                        plot_trials_wrt(names, datas, pdf, 7, col)

def plot_configs_wrt(title, names, datas, pdf, ycol):
        """ plot the test results of multiple models on the sample plot """
        colnames = datas[0].dtype.names
        # x axis -
        xlabels = names
        # y axis - loss value, loss error, #epochs, convergence speed, time
        yname = colnames[ycol]
        ylabel = yname.replace("train_", "").replace("valid_", "").replace("test_", "").replace("_", " ")
        # plot
        plt.ylabel(ylabel, fontsize = "smaller")
        plt.title(title, weight = "bold")
        bdata = []
        for data in datas:
                bdata.append(data[yname])
        plt.boxplot(bdata, labels = xlabels)
        plt.grid(True, linestyle='--')
        pdf.savefig()
        plt.close()

def plot_configs(spaths, ppath, names):
        __, datas = load_csvs(spaths, load_config_csv)
        assert(len(names) == len(datas))
        title = os.path.basename(ppath).replace(".pdf", "").replace("result_", "")
        with PdfPages(ppath) as pdf:
                for col in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
                        plot_configs_wrt(title, names, datas, pdf, col)
