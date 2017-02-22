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

activations=["act-snorm", "act-splus", "act-swave"]
for activation in activations:
        exp.add_model(("convnet4-" + activation).replace("-", "_"), (convnet4 + outlayer).replace("act-snorm", activation))

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early")

# summarize configurations
exp.summarize(trials)
