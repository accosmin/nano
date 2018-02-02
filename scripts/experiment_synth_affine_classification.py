from config import *
from experiment import *

# initialize experiment:
# - classification problem: predict the sign of a linear transformation
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_affine_classification", trials = 10)

isize = 16
osize = 8

exp.set_task(cfg.task_synth_affine_regression(isize = isize, osize = osize, noise = 0.0, count = 4000))

# loss functions
exp.add_loss("mlogistic", cfg.loss("m-logistic"))

# trainers
batch = 8
epochs = 100
patience = 100
epsilon = 1e-6
tune_epochs = 16

exp.add_trainer("batch_gd", cfg.batch_trainer("gd", epochs, patience, epsilon))
exp.add_trainer("batch_cgd", cfg.batch_trainer("cgd", epochs, patience, epsilon))
exp.add_trainer("batch_lbfgs", cfg.batch_trainer("lbfgs", epochs, patience, epsilon))

#exp.add_trainer("stoch_ag", cfg.stoch_trainer("ag", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_agfr", cfg.stoch_trainer("agfr", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_aggr", cfg.stoch_trainer("aggr", epochs, patience, epsilon, batch, tune_epochs))

exp.add_trainer("stoch_sg", cfg.stoch_trainer("sg", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_sgm", cfg.stoch_trainer("sgm", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_ngd", cfg.stoch_trainer("ngd", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_asgd", cfg.stoch_trainer("asgd", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_svrg", cfg.stoch_trainer("svrg", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_rmsprop", cfg.stoch_trainer("rmsprop", epochs, patience, epsilon, batch, tune_epochs))

#exp.add_trainer("stoch_adam", cfg.stoch_trainer("adam", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_adagrad", cfg.stoch_trainer("adagrad", epochs, patience, epsilon, batch, tune_epochs))
#exp.add_trainer("stoch_adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon, batch, tune_epochs))

# models
exp.add_model("linear", cfg.linear(imaps=isize, irows=1, icols=1, omaps=osize, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch*")
exp.summarize_by_trainers("batch*")
exp.summarize_by_trainers(".*")
