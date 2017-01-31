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

# SVHN dataset
dbdir = cfg.dbdir + "/svhn/"
os.makedirs(dbdir, exist_ok = True)

download("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", dbdir)
download("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", dbdir)
download("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", dbdir)

# MNIST dataset
dbdir = cfg.dbdir + "/mnist/"
os.makedirs(dbdir, exist_ok = True)

download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", dbdir)
download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", dbdir)
download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", dbdir)
download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", dbdir)

# STL10 dataset
dbdir = cfg.dbdir + "/stl10/"
os.makedirs(dbdir, exist_ok = True)

download("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", dbdir)

# CIFAR10 dataset
dbdir = cfg.dbdir + "/cifar10/"
os.makedirs(dbdir, exist_ok = True)

download("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", dbdir)

# CIFAR100 dataset
dbdir = cfg.dbdir + "/cifar100/"
os.makedirs(dbdir, exist_ok = True)

download("http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz", dbdir)
