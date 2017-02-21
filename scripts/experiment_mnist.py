import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.app_train,
        cfg.app_stats,
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_trainers")

# loss functions
exp.add_losses(cfg.losses(), [
        "loss_classnll"])

# criteria
exp.add_criteria(cfg.criteria(), [
        "crit_avg"])

# trainers
exp.add_trainers(cfg.trainers(), [
        "batch_gd", "batch_cgd", "batch_lbfgs",
        "stoch_sg", "stoch_sgm", "stoch_ngd", "stoch_svrg", "stoch_asgd",
        "stoch_ag", "stoch_agfr", "stoch_aggr",
        "stoch_adam", "stoch_adadelta", "stoch_adagrad", "stoch_rmsprop"])

# models
outlayer = "affine:dims=10;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

exp.add_model("convnet1", convnet1 + outlayer)
exp.add_model("convnet2", convnet2 + outlayer)
exp.add_model("convnet3", convnet3 + outlayer)
exp.add_model("convnet4", convnet4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early")

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
