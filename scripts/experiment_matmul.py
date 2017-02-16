import config
import experiment

# initialize experiment:
# - regression problem using a synthetic task
# - the model should predict the affine transformation of the multiplication of two input matrices
cfg = config.config()
exp = experiment.experiment(
        cfg.app_train,
        cfg.app_stats,
        cfg.get_task_synth_matmul(irows = 23, icols = 27, count = 10000, noise = 1e-4),
        cfg.expdir + "/matmul/eval_trainers")

# loss functions
exp.add_losses(cfg.losses(), [
        "loss_cauchy"])

# criteria
exp.add_criteria(cfg.criteria(), [
        "crit_avg"])

# trainers
exp.add_trainers(cfg.trainers(), [
        "batch_gd", "batch_cgd", "batch_lbfgs",
        "stoch_sg", "stoch_sgm", "stoch_ngd", "stoch_svrg", "stoch_asgd",
        "stoch_ag", "stoch_agfr", "stoch_aggr",
        "stoch_adam", "stoch_adadelta", "stoch_adagrad"])

# models
outlayer = "affine:dims=" + str(23 * 23) + ";act-snorm;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=100;act-snorm;"
mlp2 = mlp1 + "affine:dims=100;act-snorm;"
mlp3 = mlp2 + "affine:dims=100;act-snorm;"

exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("mlp3", mlp3 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, cfg.policies().get("stop_early"))

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
