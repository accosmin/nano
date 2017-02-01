import config
import experiment

# initialize experiment:
# - single-class classification problem using a synthetic task
# - the model should predict the digit of a grayscale image
cfg = config.config()

task = "--task mnist --task-params dir=" + cfg.dbdir + "/mnist"
exp = experiment.experiment(cfg.app_train, cfg.app_stats, task, cfg.expdir + "/mnist/eval_trainers")

# loss functions
losses = ["loss_classnll"]
for name in losses:
        exp.add_loss(name, cfg.losses.get(name))

# criteria
criteria = ["crit_avg"]
for name in criteria:
        exp.add_criterion(name, cfg.criteria.get(name))

# models
outlayer = "affine:dims=10;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-splus;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-splus;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-splus;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-splus;"

exp.add_model("convnet1", convnet1 + outlayer)
exp.add_model("convnet2", convnet2 + outlayer)
exp.add_model("convnet3", convnet3 + outlayer)
exp.add_model("convnet4", convnet4 + outlayer)

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
epochs = 20
exp.run_all(trials, epochs, cfg.policies.get("stop_early"))

# compare configurations
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
