from experiment import *

# initialize experiment:
# - regression problem: predict the output of an affine transformation
cfg = config()
exp = experiment(cfg.expdir + "/synth_affine_regression", trials = 10)

isize = 32
osize = 32

exp.set_task(cfg.task_synth_affine_regression(isize = isize, osize = osize, noise = 0.0, count = 10000))

# loss functions
exp.add_loss("cauchy", cfg.loss("cauchy"))
exp.add_loss("square", cfg.loss("square"))

# enhancers
exp.add_enhancer("default", cfg.enhancer("default"))

# trainers
batch = 1
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
exp.add_trainer("stoch_ngd", cfg.stoch_trainer("ngd", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_asgd", cfg.stoch_trainer("asgd", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_svrg", cfg.stoch_trainer("svrg", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_rmsprop", cfg.stoch_trainer("rmsprop", epochs, patience, epsilon, batch, tune_epochs))

exp.add_trainer("stoch_adam", cfg.stoch_trainer("adam", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_adagrad", cfg.stoch_trainer("adagrad", epochs, patience, epsilon, batch, tune_epochs))
exp.add_trainer("stoch_adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon, batch, tune_epochs))

# models
exp.add_model("linear", cfg.linear(imaps=isize, irows=1, icols=1, omaps=osize, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_models(".*")
exp.summarize_by_trainers("stoch*")
exp.summarize_by_trainers("batch*")
exp.summarize_by_trainers(".*")
