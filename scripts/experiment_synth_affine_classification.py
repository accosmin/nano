from config import *
from experiment import *

# initialize experiment:
# - classification problem: predict the sign of a linear transformation
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_affine_classification", trials = 10)

isize = 16
osize = 8

exp.set_task(cfg.task_synth_affine_classification(isize = isize, osize = osize, noise = 0.0, count = 4000))

# loss functions
exp.add_loss("mlogistic", cfg.loss("m-logistic"))

# trainers
batch = 8
epochs = 100
patience = 100
epsilon = 1e-6
tune_epochs = 10

exp.add_trainer("gd", cfg.batch_trainer("gd", epochs, patience, epsilon))
exp.add_trainer("cgd", cfg.batch_trainer("cgd", epochs, patience, epsilon))
exp.add_trainer("lbfgs", cfg.batch_trainer("lbfgs", epochs, patience, epsilon))

exp.add_trainer("ag", cfg.stoch_trainer("ag", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("agfr", cfg.stoch_trainer("agfr", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("aggr", cfg.stoch_trainer("aggr", epochs, patience, epsilon, batch, tune_epochs))

exp.add_trainer("sg", cfg.stoch_trainer("sg", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("sgm", cfg.stoch_trainer("sgm", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("ngd", cfg.stoch_trainer("ngd", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("asgd", cfg.stoch_trainer("asgd", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("svrg", cfg.stoch_trainer("svrg", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("rmsprop", cfg.stoch_trainer("rmsprop", epochs, patience, epsilon, batch, tune_epochs))

exp.add_trainer("adam", cfg.stoch_trainer("adam", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("adagrad", cfg.stoch_trainer("adagrad", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon, batch, tune_epochs))

# models
exp.add_model("linear", cfg.linear(imaps=isize, irows=1, icols=1, omaps=osize, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "ag|agfr|aggr|sg|sgm|ngd|asgd|svrg|rmsprop|adam|adagrad|adadelta")
exp.summarize_by_trainers("batch", "gd|cgd|lbfgs")
exp.summarize_by_trainers("all", ".*")
