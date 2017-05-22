import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_iterators")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")
exp.add_iterator("noise", "noise=0.05", "noise05")
exp.add_iterator("noise", "noise=0.10", "noise10")
exp.add_iterator("noise", "noise=0.20", "noise20")
exp.add_iterator("noise", "noise=0.50", "noise50")
exp.add_iterator("noise", "noise=0.99", "noise99")
exp.add_iterator("warp")

# trainers
batch_params = "epochs=100,patience=32,epsilon=1e-6"
stoch_params = "epochs=100,patience=32,epsilon=1e-6,min_batch=32,max_batch=256"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
outlayer = "affine:dims=10;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"

exp.add_model("convnet4", convnet4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare iterators
for tname, mname, lname in [(x, y, z) for x in exp.trainers for y in exp.models for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, tname, ".*", lname, ".state"),
                        exp.path(trial, mname, tname, None, lname, ".pdf"))

        exp.summarize(trials, mname, tname, ".*", lname,
                exp.path(None, mname, tname, None, lname, ".log"),
                exp.path(None, mname, tname, None, lname, ".csv"))
