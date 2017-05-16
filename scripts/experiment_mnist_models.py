import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_models")

# loss functions
exp.add_losses([
        "loss_classnll"])

# iterators
exp.add_iterators([
        "default"])

# trainers
exp.add_trainers([
        "stoch_svrg"])

# models
outlayer = "affine:dims=10;act-snorm;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-snorm;"
mlp2 = mlp1 + "affine:dims=128;act-snorm;"
mlp3 = mlp2 + "affine:dims=128;act-snorm;"
mlp4 = mlp3 + "affine:dims=128;act-snorm;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=4,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=4,drow=1,dcol=1;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)
exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("mlp3", mlp3 + outlayer)
exp.add_model("mlp4", mlp4 + outlayer)
exp.add_model("convnet1", convnet1 + outlayer)
exp.add_model("convnet2", convnet2 + outlayer)
exp.add_model("convnet3", convnet3 + outlayer)
exp.add_model("convnet4", convnet4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for iname in exp.iterators:
                for lname in exp.losses:
                        # compare mlps
                        exp.plot_many(
                                exp.filter(trial, "mlp*", ".*", iname, lname, ".state"),
                                exp.get_path(trial, "mlp", "", iname, lname, ".pdf"))

                        # compare convnets
                        exp.plot_many(
                                exp.filter(trial, "convnet*", ".*", iname, lname, ".state"),
                                exp.get_path(trial, "convnet", "", iname, lname, ".pdf"))

                        # compare all models
                        exp.plot_many(
                                exp.filter(trial, ".*", ".*", iname, lname, ".state"),
                                exp.get_path(trial, "", "", iname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
