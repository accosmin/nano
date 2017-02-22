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
        cfg.expdir + "/mnist/eval_activations")

# loss functions
exp.add_losses(cfg.losses(), [
        "loss_classnll"])

# criteria
exp.add_criteria(cfg.criteria(), [
        "crit_avg"])

# trainers
exp.add_trainers(cfg.trainers(), [
        "stoch_svrg"])

# models
outlayer = "affine:dims=10;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

for activation in ["act-snorm", "act-splus", "act-swave", "act-unit", "act-tanh", "act-sin"]:
        name = ("convnet4-" + activation).replace("-", "_")
        params = (convnet4 + outlayer).replace("act-snorm", activation)
        exp.add_model(name, params)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for mname in exp.models:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                # compare all activation functions
                                exp.plot_many(
                                        exp.filter(trial, mname, ".*", cname, lname, ".state"),
                                        exp.get_path(trial, mname, "", cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
