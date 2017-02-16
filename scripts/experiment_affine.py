import config
import experiment

# initialize experiment:
# - regression problem using a synthetic task
# - the model should predict an affine mapping of the input vector
cfg = config.config()
exp = experiment.experiment(
        cfg.app_train,
        cfg.app_stats,
        cfg.get_task_synth_affine(isize = 100, osize = 10, count = 10000, noise = 1e-4),
        cfg.expdir + "/affine/eval_trainers")

# loss functions
losses = ["loss_cauchy"]
for name in losses:
        exp.add_loss(name, cfg.losses.get(name))

# criteria
criteria = ["crit_avg"]
for name in criteria:
        exp.add_criterion(name, cfg.criteria.get(name))

# models
outlayer = "affine:dims=10;"

mlp0 = "--model forward-network --model-params "

exp.add_model("mlp0", mlp0 + outlayer)

# trainers
trainers = []
trainers += ["batch_gd", "batch_cgd", "batch_lbfgs"]
trainers += ["stoch_sg", "stoch_sgm", "stoch_ngd", "stoch_svrg", "stoch_asgd"]
trainers += ["stoch_ag", "stoch_agfr", "stoch_aggr"]
trainers += ["stoch_adam", "stoch_adadelta", "stoch_adagrad"]
for name in trainers:
        exp.add_trainer(name, cfg.trainers.get(name))

# train all configurations
trials = 10
epochs = 100
exp.run_all(trials, epochs, cfg.policies.get("stop_early"))

# compare configurations
for trial in range(trials):
        for mname in exp.models:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                stoch_spaths = exp.filter(trial, mname, "stoch*", cname, lname, ".state")
                                batch_spaths = exp.filter(trial, mname, "batch*", cname, lname, ".state")
                                all_spaths = exp.filter(trial, mname, ".*", cname, lname, ".state")

                                # compare stochastic trainers
                                exp.plot_many(stoch_spaths, exp.get_path(trial, mname, "stoch", cname, lname, ".pdf"))

                                # compare batch trainers
                                exp.plot_many(batch_spaths, exp.get_path(trial, mname, "batch", cname, lname, ".pdf"))

                                # compare all trainers
                                exp.plot_many(all_spaths, exp.get_path(trial, mname, "all", cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
