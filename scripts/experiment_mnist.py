import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist")

# loss functions
exp.add_loss("slogistic")
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")
exp.add_iterator("noise", "noise=0.05", "noise05")
exp.add_iterator("noise", "noise=0.10", "noise10")
exp.add_iterator("noise", "noise=0.20", "noise20")
exp.add_iterator("noise", "noise=0.50", "noise50")
exp.add_iterator("noise", "noise=0.99", "noise99")

# trainers
batch_params = "epochs=10,policy=stop_early,patience=32,epsilon=1e-6"
stoch_params = "epochs=10,policy=stop_early,patience=32,epsilon=1e-6,min_batch=32,max_batch=256"

exp.add_trainers("stoch_adadelta", stoch_params)

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
exp.run_all(trials = trials)

# compare models
for tname, iname, lname in [(x, y, z) for x in exp.trainers for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, "mlp*", ".*", iname, lname, ".state"),
                        exp.path(trial, "mlp", "", iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, "convnet*", ".*", iname, lname, ".state"),
                        exp.path(trial, "convnet", "", iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, ".*", ".*", iname, lname, ".state"),
                        exp.path(trial, "", "", iname, lname, ".pdf"))

# compare iterators
for tname, mname, lname in [(x, y, z) for x in exp.trainers for y in exp.models for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, tname, ".*", lname, ".state"),
                        exp.path(trial, mname, tname, "", lname, ".pdf"))

# compare losses
for tname, mname, iname in [(x, y, z) for x in exp.trainers for y in exp.models for z in exp.iterators]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, tname, iname, ".*", ".state"),
                        exp.path(trial, mname, tname, iname, "", "*.pdf"))

# summarize configurations
exp.summarize(trials)