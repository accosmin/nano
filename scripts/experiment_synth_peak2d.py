from config import *
from experiment import *

# initialize experiment:
# - regression problem: predict the position of a peak in an image
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_peak2d", trials = 10)

exp.set_task(cfg.task_synth_peak2d(irows = 16, icols = 16, noise = 0.0, count = 4000))

# loss functions
exp.add_loss("cauchy", cfg.loss("cauchy"))

# trainers
epochs = 100
patience = 100
epsilon = 1e-6

exp.add_trainer("gd", cfg.batch_trainer("gd", epochs, patience, epsilon))
exp.add_trainer("cgd", cfg.batch_trainer("cgd", epochs, patience, epsilon))
exp.add_trainer("lbfgs", cfg.batch_trainer("lbfgs", epochs, patience, epsilon))

exp.add_trainer("ag", cfg.stoch_trainer("ag", epochs, patience, epsilon))
exp.add_trainer("agfr", cfg.stoch_trainer("agfr", epochs, patience, epsilon))
exp.add_trainer("aggr", cfg.stoch_trainer("aggr", epochs, patience, epsilon))

exp.add_trainer("sg", cfg.stoch_trainer("sg", epochs, patience, epsilon))
exp.add_trainer("sgm", cfg.stoch_trainer("sgm", epochs, patience, epsilon))
exp.add_trainer("ngd", cfg.stoch_trainer("ngd", epochs, patience, epsilon))
exp.add_trainer("asgd", cfg.stoch_trainer("asgd", epochs, patience, epsilon))
exp.add_trainer("svrg", cfg.stoch_trainer("svrg", epochs, patience, epsilon))
exp.add_trainer("rmsprop", cfg.stoch_trainer("rmsprop", epochs, patience, epsilon))

exp.add_trainer("adam", cfg.stoch_trainer("adam", epochs, patience, epsilon))
exp.add_trainer("adagrad", cfg.stoch_trainer("adagrad", epochs, patience, epsilon))
exp.add_trainer("amsgrad", cfg.stoch_trainer("amsgrad", epochs, patience, epsilon))
exp.add_trainer("adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon))

# models
mlp0 = []
mlp1 = mlp0 + [128,1,1]
mlp2 = mlp1 + [256,1,1]
mlp3 = mlp2 + [512,1,1]
mlp4 = mlp3 + [1024,1,1]

cnn1 = [32,5,5,1,2,2]
cnn2 = cnn1 + [64,3,3,1,1,1]
cnn3 = cnn2 + [128,3,3,1,1,1]
cnn4 = cnn3 + [256,3,3,1,1,1]

exp.add_model("mlp0", cfg.mlp(mlp0, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("mlp1", cfg.mlp(mlp1, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("mlp2", cfg.mlp(mlp2, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("mlp3", cfg.mlp(mlp3, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("mlp4", cfg.mlp(mlp4, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))

exp.add_model("cnn1", cfg.cnn(cnn1, mlp0, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("cnn2", cfg.cnn(cnn2, mlp0, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("cnn3", cfg.cnn(cnn3, mlp0, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))
exp.add_model("cnn4", cfg.cnn(cnn4, mlp0, imaps=1, irows=16, icols=16, omaps=2, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "ag|agfr|aggr|sg|sgm|ngd|asgd|svrg|rmsprop|adam|adagrad|amsgrad|adadelta")
exp.summarize_by_trainers("batch", "gd|cgd|lbfgs")
exp.summarize_by_trainers("all", ".*")
