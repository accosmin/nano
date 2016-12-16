import config
import experiment

# initialize experiment:
# - single-class classification problem using a synthetic task
# - the model should predict the digit of a synthetic image
task = "--task charset --task-params type=digit,color=rgb,irows=16,icols=16,count=10000"

cfg = config.config()
exp = experiment.experiment(cfg.app_train, cfg.app_stats, task, cfg.expdir + "/charset/eval_trainers")

# loss functions
losses = "loss_classnll"
for name in losses.split():
        exp.add_loss(name, cfg.losses.get(name))

# criteria
criteria = "crit_avg"
for name in criteria.split():
        exp.add_criterion(name, cfg.criteria.get(name))

# models
outlayer = "affine:dims=10;act-snorm;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=10;act-snorm;"
mlp2 = mlp1 + "affine:dims=10;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)
exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("convnet1", convnet1 + outlayer)
exp.add_model("convnet2", convnet2 + outlayer)
exp.add_model("convnet3", convnet3 + outlayer)

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
epochs = 100
exp.run_all(trials, epochs, cfg.policies.get("stop_early"))

# compare configurations
for trial in range(trials):
        for mname in exp.models:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                stoch_spaths = []
                                batch_spaths = []
                                all_spaths = []
                                for tname in exp.trainers:
                                        spath = exp.get_path(trial, mname, tname, cname, lname, ".state")
                                        if tname.find("stoch") < 0:
                                                batch_spaths.append(spath)
                                        else:
                                                stoch_spaths.append(spath)
                                        all_spaths.append(spath)

                                # compare stochastic trainers
                                exp.plot_many(stoch_spaths, exp.get_path(trial, mname, "stoch", cname, lname, ".pdf"))

                                # compare batch trainers
                                exp.plot_many(batch_spaths, exp.get_path(trial, mname, "batch", cname, lname, ".pdf"))

                                # compare all trainers
                                exp.plot_many(all_spaths, exp.get_path(trial, mname, "all", cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
