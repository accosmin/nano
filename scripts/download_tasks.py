import os
import re
import config
import urllib3
import argparse

def download(url, dbdir):
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

def mkdir(dbname):
        cfg = config.config()
        dbdir = cfg.dbdir + "/" + dbname + "/"
        os.makedirs(dbdir, exist_ok = True)
        return dbdir

def download_iris():
        dbdir = mkdir("iris")
        download("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", dbdir)

def download_wine():
        dbdir = mkdir("wine")
        download("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", dbdir)

def download_svhn():
        dbdir = mkdir("svhn")
        download("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", dbdir)
        download("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", dbdir)
        download("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", dbdir)

def download_mnist():
        dbdir = mkdir("mnist")
        download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dbdir)
        download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dbdir)
        download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dbdir)
        download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dbdir)

def download_cifar10():
        dbdir = mkdir("cifar10")
        download("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", dbdir)

def download_cifar100():
        dbdir = mkdir("cifar100")
        download("http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz", dbdir)

def download_fashion_mnist():
        dbdir = mkdir("fashion-mnist")
        download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", dbdir)
        download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", dbdir)
        download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", dbdir)
        download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", dbdir)

def download_tasks(task_name_regex):
        tasks = {
                "iris": download_iris,
                "wine": download_wine,
                "svhn": download_svhn,
                "mnist": download_mnist,
                "cifar10": download_cifar10,
                "cifar100": download_cifar100,
                "fashion-mnist": download_fashion_mnist
        }
        for name, func in tasks.items():
                if re.match(task_name_regex, name):
                        func()

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="download tasks")
        parser.add_argument("--tasks", action="store", help="regex of the task names to download", type=str)

        args = parser.parse_args()
        download_tasks(args.tasks)
