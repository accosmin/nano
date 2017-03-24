import config
import experiment

# initialize experiment:
# - single-class classification problem using a synthetic task
# - the model should predict the digit of a synthetic image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_synth_charset(ctype = "digit", color = "rgb", irows = 18, icols = 18, count = 10000),
        cfg.expdir + "/charset/eval_trainers")

# loss functions
exp.add_losses([
        "loss_classnll"])

# criteria
exp.add_criteria([
        "crit_avg"])

# trainers
exp.add_trainers([
        "batch_gd", "batch_cgd", "batch_lbfgs",
        "stoch_sg", "stoch_sgm", "stoch_ngd", "stoch_svrg", "stoch_asgd",
        "stoch_ag", "stoch_agfr", "stoch_aggr",
        "stoch_adam", "stoch_adadelta", "stoch_adagrad", "stoch_rmsprop"])

# models
mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-snorm;"
mlp2 = mlp1 + "affine:dims=64;act-snorm;"
mlp3 = mlp2 + "affine:dims=32;act-snorm;"
mlp4 = mlp3 + "affine:dims=32;act-snorm;"
mlp5 = mlp4 + "affine:dims=32;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet5 = convnet4 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

outlayer = "affine:dims=10;act-snorm;"

#exp.add_model("mlp0", mlp0 + outlayer)
#exp.add_model("mlp1", mlp1 + outlayer)
#exp.add_model("mlp2", mlp2 + outlayer)
#exp.add_model("mlp3", mlp3 + outlayer)
#exp.add_model("mlp4", mlp4 + outlayer)
exp.add_model("mlp5", mlp5 + outlayer)
#exp.add_model("convnet1", convnet1 + outlayer)
#exp.add_model("convnet2", convnet2 + outlayer)
#exp.add_model("convnet3", convnet3 + outlayer)
#exp.add_model("convnet4", convnet4 + outlayer)
exp.add_model("convnet5", convnet5 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for mname in exp.models:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                # compare stochastic trainers
                                exp.plot_many(
                                        exp.filter(trial, mname, "stoch*", cname, lname, ".state"),
                                        exp.get_path(trial, mname, "stoch", cname, lname, ".pdf"))

                                # compare batch trainers
                                exp.plot_many(
                                        exp.filter(trial, mname, "batch*", cname, lname, ".state"),
                                        exp.get_path(trial, mname, "batch", cname, lname, ".pdf"))

                                # compare all trainers
                                exp.plot_many(
                                        exp.filter(trial, mname, ".*", cname, lname, ".state"),
                                        exp.get_path(trial, mname, "all", cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
