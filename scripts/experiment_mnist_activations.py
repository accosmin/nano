import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit represented by an image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_activations")

# loss functions
exp.add_losses([
        "loss_classnll"])

# criteria
exp.add_criteria([
        "crit_avg"])

# trainers
exp.add_trainers([
        "stoch_svrg"])

# models
outlayer = "affine:dims=10;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=128,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convnet2 = convnet1 + "conv:dims=128,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=128,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=128,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

for activation in ["snorm", "tanh", "sin", "ewave:alpha=1", "ewave:alpha=2", "ewave:alpha=3", "ewave:alpha=4", "pwave"]:
        name = ("convnet4-act-" + activation).replace("-", "_").replace(":alpha=", "")
        params = (convnet4 + outlayer).replace("act-snorm", "act-" + activation)
        exp.add_model(name, params)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 1000, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for tname in exp.trainers:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                # compare all activation functions
                                exp.plot_many(
                                        exp.filter(trial, ".*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "", tname, cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
