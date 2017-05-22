import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_losses")

# loss functions
exp.add_loss("classnll")
exp.add_loss("slogistic")
exp.add_loss("sexponential")

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

exp.add_model("convnet4", convnet4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare losses
for tname, mname, iname in [(x, y, z) for x in exp.trainers for y in exp.models for z in exp.iterators]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, tname, iname, ".*", ".state"),
                        exp.path(trial, mname, tname, iname, None, ".pdf"))

        exp.summarize(trials, mname, tname, iname, ".*",
                exp.path(None, mname, tname, iname, None, ".log"),
                exp.path(None, mname, tname, iname, None, ".csv"))
