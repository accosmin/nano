import experiment as nano

cfg = nano.config()
print(cfg.dbdir)
print(cfg.expdir)
print(cfg.app_trainer)
print("losses = ", cfg.losses)

# initialize experiment
task = "--task affine --task-params isize=100,osize=10,count=10000,noise=1e-4"
outdir = cfg.expdir + "/affine/eval_trainers"

exp = nano.experiment(task, outdir)

# loss functions
exp.add_loss("loss_cauchy", cfg.losses.get("loss_cauchy"))
exp.add_loss("loss_square", cfg.losses.get("loss_square"))

# criteria
exp.add_criterion("crit_avg", cfg.criteria.get("crit_avg"))
exp.add_criterion("crit_max", cfg.criteria.get("crit_max"))

# models
outlayer = "affine:dims=10;act-snorm;"

mlp0 = "--model forward-network --model-params "
#mlp1 = mlp0 + "affine:dims=100;act-snorm;"
#mlp2 = mlp1 + "affine:dims=100;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)

# trainers
fn_make_trainers "stop_early"
trainers=${trainers}" batch_gd batch_cgd batch_lbfgs"
trainers=${trainers}" stoch_sg stoch_sgm stoch_ngd stoch_svrg stoch_asgd"
trainers=${trainers}" stoch_ag stoch_agfr stoch_aggr"
trainers=${trainers}" stoch_adam stoch_adadelta stoch_adagrad"

# train all configurations
fn_train "${outdir}" "${task}" "${models}" "${trainers}" "${criteria}" "${losses}"

trials = 10
epochs = 100
policy = cfg.policies.get("stop_early")

# compare models
