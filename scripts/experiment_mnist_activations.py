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

# iterators
exp.add_iterators([
        "default"])

# trainers
exp.add_trainers([
        "stoch_adadelta"])

# models
outlayer = "affine:dims=10;act-snorm;"

convnet = "--model forward-network --model-params "
convnet = convnet + "conv:dims=128,rows=7,cols=7,conn=1,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=7,cols=7,conn=16,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=5,cols=5,conn=16,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=5,cols=5,conn=16,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=3,cols=3,conn=16,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=3,cols=3,conn=16,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=128,rows=3,cols=3,conn=16,drow=1,dcol=1;act-snorm;"

for activation in ["snorm", "tanh", "sin", "pwave"]:
        name = ("convnet-act-" + activation).replace("-", "_")
        params = (convnet + outlayer).replace("act-snorm", "act-" + activation)
        exp.add_model(name, params)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 1000, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for tname in exp.trainers:
                for iname in exp.iterators:
                        for lname in exp.losses:
                                # compare all activation functions
                                exp.plot_many(
                                        exp.filter(trial, ".*", tname, iname, lname, ".state"),
                                        exp.get_path(trial, "", tname, iname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
