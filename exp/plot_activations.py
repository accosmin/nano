import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def gen_x(xmin, xmax, count):
        xs = []
        dx = (xmax - xmin) / count
        for k in range(count + 1):
                x = xmin + k * dx
                xs.append(x)
        return xs

def gen_ewave(xs, alpha, ys, labels):
        ys.append(list(map(lambda x : 2 * alpha * x / (math.exp(-alpha * x) + math.exp(+alpha * x)), xs)))
        labels.append("{}*x/(exp(-{}*x) + exp(+{}*x))".format(alpha, alpha, alpha))

def gen_pwave(xs, degree, ys, labels):
        ys.append(list(map(lambda x : x / (1 + x**degree), xs)))
        labels.append("x/(1+x^{})".format(degree))

def gen_snorm(xs, ys, labels):
        ys.append(list(map(lambda x : x / math.sqrt(1 + x**2), xs)))
        labels.append("x/sqrt(1+x^2)")

def gen_tanh(xs, ys, labels):
        ys.append(list(map(lambda x : math.tanh(x), xs)))
        labels.append("tanh(x)")

def gen_sin(xs, ys, labels):
        ys.append(list(map(lambda x : math.sin(x), xs)))
        labels.append("sin(x)")

def plot(ppath, title, xs, ys, labels, styles):
        with PdfPages(ppath) as pdf:
                plt.xlabel("x", fontsize = "smaller")
                plt.ylabel("y", fontsize = "smaller")
                plt.title(title, weight = "bold")
                for y, label, style in zip(ys, labels, styles):
                        plt.plot(xs, y, style, label = label)
                plt.legend(fontsize = "smaller", loc = "upper left")
                plt.grid(True)
                pdf.savefig()
                plt.close()

def plot_pwaves(ppath, xmin = -10, xmax = 10):
        xs = gen_x(xmin, xmax, 1000)
        ys = []
        labels = []
        gen_pwave(xs, 2, ys, labels)
        gen_pwave(xs, 4, ys, labels)
        gen_pwave(xs, 6, ys, labels)
        gen_pwave(xs, 8, ys, labels)
        plot(ppath, "polynomial wave activations", xs, ys, labels, ["r-", "g-", "b-", "k-"])

def plot_ewaves(ppath, xmin = -10, xmax = 10):
        xs = gen_x(xmin, xmax, 1000)
        ys = []
        labels = []
        gen_ewave(xs, 1, ys, labels)
        gen_ewave(xs, 2, ys, labels)
        gen_ewave(xs, 3, ys, labels)
        gen_ewave(xs, 4, ys, labels)
        plot(ppath, "exponential wave activations", xs, ys, labels, ["r-", "g-", "b-", "k-"])

def plot_snorms(ppath, xmin = -10, xmax = 10):
        xs = gen_x(xmin, xmax, 1000)
        ys = []
        labels = []
        gen_snorm(xs, ys, labels)
        gen_tanh(xs, ys, labels)
        gen_sin(xs, ys, labels)
        gen_pwave(xs, 2, ys, labels)
        plot(ppath, "[-1, +1] normalization activations", xs, ys, labels, ["r-", "g-", "b-", "k-"])

plot_pwaves("activations_pwave.pdf")
plot_ewaves("activations_ewave.pdf")
plot_snorms("activations_snorm.pdf")
