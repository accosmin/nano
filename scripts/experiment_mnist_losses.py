import config
import experiment
import models_mnist as models

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
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,min_batch=32,max_batch=256"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("convnet5", models.convnet5 + models.outlayer)

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
