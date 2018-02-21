from config import *
from experiment import *

# initialize experiment:
# - regression problem: predict the output of an affine transformation
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_affine", trials = 10)

exp.set_task(cfg.task_synth_affine(isize = 16, osize = 8, noise = 0.0, count = 4000))

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
exp.add_model("linear", cfg.linear(imaps=16, irows=1, icols=1, omaps=8, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "ag|agfr|aggr|sg|sgm|ngd|asgd|svrg|rmsprop|adam|adagrad|amsgrad|adadelta")
exp.summarize_by_trainers("batch", "gd|cgd|lbfgs")
exp.summarize_by_trainers("all", ".*")
