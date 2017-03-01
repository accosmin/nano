import os
import config
import urllib3

cfg = config.config()

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

def download_stl10():
        dbdir = mkdir("stl10")
        download("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", dbdir)

def download_cifar10():
        dbdir = mkdir("cifar10")
        download("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", dbdir)

def download_cifar100():
        dbdir = mkdir("cifar100")
        download("http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz", dbdir)

download_iris()
download_wine()
download_svhn()
download_mnist()
download_stl10()
download_cifar10()
download_cifar100()
