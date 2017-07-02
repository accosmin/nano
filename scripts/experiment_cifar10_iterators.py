import config
import experiment
import models_cifar10 as models

# initialize experiment:
# - single-class classification problem using the CIFAR-10 dataset
# - the model should predict the object of a RGB image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/cifar10/eval_iterators")

# loss functions
exp.add_loss("slogistic")

# iterators
exp.add_iterator("default")
exp.add_iterator("noise", "noise=0.10", "noise10")
exp.add_iterator("noise", "noise=0.20", "noise20")
exp.add_iterator("noise", "noise=0.40", "noise40")
exp.add_iterator("noise", "noise=0.80", "noise80")
exp.add_iterator("noclass", "ratio=0.10,noise=0.00", "noclass10")
exp.add_iterator("noclass", "ratio=0.20,noise=0.00", "noclass20")
exp.add_iterator("noclass", "ratio=0.40,noise=0.00", "noclass40")
exp.add_iterator("noclass", "ratio=0.80,noise=0.00", "noclass80")
exp.add_iterator("warp")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("convnet9", models.convnet9 + models.outlayer)

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
