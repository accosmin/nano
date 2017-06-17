import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_minibatch/")

# loss functions
exp.add_loss("slogistic")

# iterators
exp.add_iterator("default")

# trainers
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,min_batch={},max_batch={}"
minibatch_name = "minibatch{}to{}"

for size in [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024], [32, 1024], [32, 16 * 1024]]:
        exp.add_trainer("stoch_adadelta", stoch_params.format(size[0], size[1]), minibatch_name.format(size[0], size[1]))

# models
exp.add_model("convnet5", models.convnet5 + models.outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare trainers
for mname, iname, lname in [(x, y, z) for x in exp.models for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, ".*", iname, lname, ".state"),
                        exp.path(trial, mname, None, iname, lname, ".pdf"))

        exp.summarize(trials, mname, ".*", iname, lname,
                exp.path(None, mname, None, iname, lname, ".log"),
                exp.path(None, mname, None, iname, lname, ".csv"))
