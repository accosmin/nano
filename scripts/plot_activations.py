import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def gen_poly(xmin, xmax, count, degree):
        xs = []
        ys = []
        dx = (xmax - xmin) / count
        for k in range(count + 1):
                x = xmin + k * dx
                y = x / (1 + x**degree)
                xs.append(x)
                ys.append(y)
        return xs, ys

def gen_sigm(xmin, xmax, count, alpha):
        xs = []
        ys = []
        dx = (xmax - xmin) / count
        for k in range(count + 1):
                x = xmin + k * dx
                y = 2 * alpha * x / (math.exp(-alpha * x) + math.exp(+alpha * x))
                xs.append(x)
                ys.append(y)
        return xs, ys

def plot_poly(xmin, xmax, degree, style):
        xs, ys = gen_poly(xmin, xmax, 1000, degree)
        plt.plot(xs, ys, style, label = "x/(1+x^{0})".format(degree))

def plot_sigm(xmin, xmax, alpha, style):
        xs, ys = gen_sigm(xmin, xmax, 1000, alpha)
        plt.plot(xs, ys, style, label = "2*{0}*x/(exp(-{1}*x) + exp(+{2}*x))".format(alpha, alpha, alpha))

def plot_polys(ppath, xmin = -10, xmax = 10):
        with PdfPages(ppath) as pdf:
                plt.xlabel("x", fontsize = "smaller")
                plt.ylabel("y", fontsize = "smaller")
                plt.title("polynomial waves", weight = "bold")
                plot_poly(xmin, xmax, 2, "r-")
                plot_poly(xmin, xmax, 4, "g-")
                plot_poly(xmin, xmax, 6, "b-")
                plot_poly(xmin, xmax, 8, "k-")
                plt.legend(fontsize = "smaller", loc = "upper left")
                plt.grid(True)
                pdf.savefig()
                plt.close()

def plot_sigms(ppath, xmin = -10, xmax = 10):
        with PdfPages(ppath) as pdf:
                plt.xlabel("x", fontsize = "smaller")
                plt.ylabel("y", fontsize = "smaller")
                plt.title("sigmoid waves", weight = "bold")
                plot_sigm(xmin, xmax, 1, "r-")
                plot_sigm(xmin, xmax, 2, "g-")
                plot_sigm(xmin, xmax, 3, "b-")
                plot_sigm(xmin, xmax, 4, "k-")
                plt.legend(fontsize = "smaller", loc = "upper left")
                plt.grid(True)
                pdf.savefig()
                plt.close()

plot_polys("activations_poly.pdf")
plot_sigms("activations_sigm.pdf")
