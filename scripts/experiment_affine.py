import config
import experiment

# initialize experiment
task = "--task affine --task-params isize=100,osize=10,count=10000,noise=1e-4"

cfg = config.config()
exp = experiment.experiment(cfg.app_train, task, cfg.expdir + "/affine/eval_trainers")

# loss functions
losses = "loss_cauchy loss_square"
for name in losses.split():
        exp.add_loss(name, cfg.losses.get(name))

# criteria
criteria = "crit_avg crit_max"
for name in criteria.split():
        exp.add_criterion(name, cfg.criteria.get(name))

# models
outlayer = "affine:dims=10;act-snorm;"

mlp0 = "--model forward-network --model-params "
#mlp1 = mlp0 + "affine:dims=100;act-snorm;"
#mlp2 = mlp1 + "affine:dims=100;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)

# trainers
trainers = ""
trainers += "batch_gd batch_cgd batch_lbfgs "
trainers += "stoch_sg stoch_sgm stoch_ngd stoch_svrg stoch_asgd "
trainers += "stoch_ag stoch_agfr stoch_aggr "
trainers += "stoch_adam stoch_adadelta stoch_adagrad "
for name in trainers.split():
        exp.add_trainer(name, cfg.trainers.get(name))

# train all configurations
trials = 10
epochs = 10
exp.run_all(trials, epochs, cfg.policies.get("stop_early"))

# compare models
