import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_activations/")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

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

for activation in ["snorm", "tanh", "sin", "pwave"]:
        name = ("convnet4-act-" + activation).replace("-", "_")
        params = (convnet4 + outlayer).replace("act-snorm", "act-" + activation)
        exp.add_model(name, params)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare models
for tname, iname, lname in [(x, y, z) for x in exp.trainers for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, ".*", tname, iname, lname, ".state"),
                        exp.path(trial, None, tname, iname, lname, ".pdf"))

        exp.summarize(trials, ".*", tname, iname, lname,
                exp.path(None, None, tname, iname, lname, ".log"),
                exp.path(None, None, tname, iname, lname, ".csv"))
