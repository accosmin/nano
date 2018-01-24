import os
import re
import config
import urllib3
import argparse

class downloader():
        """ utility to manage downloading well-known datasets (aka tasks)
        """
        def __init__(self):
                pass

        def get(self, url, dbdir):
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
                cfg = config.config()
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

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="command line wrapper over the experimentation scripts")
        parser.add_argument("--tasks", action="store", help="regex of the task names to download", type=str)

        args = parser.parse_args()

        if args.tasks:
                getter = downloader()
                if re.match(args.tasks, "iris"): getter.get_iris()
                if re.match(args.tasks, "wine"): getter.get_wine()
                if re.match(args.tasks, "svhn"): getter.get_svhn()
                if re.match(args.tasks, "mnist"): getter.get_mnist()
                if re.match(args.tasks, "cifar10"): getter.get_cifar10()
                if re.match(args.tasks, "cifar100"): getter.get_cifar100()
                if re.match(args.tasks, "fashion-mnist"): getter.get_fashion_mnist()

