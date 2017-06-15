import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_models/")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,min_batch=32,max_batch=256"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("mlp0", models.mlp0 + models.outlayer)
exp.add_model("mlp1", models.mlp1 + models.outlayer)
exp.add_model("mlp2", models.mlp2 + models.outlayer)
exp.add_model("mlp3", models.mlp3 + models.outlayer)
exp.add_model("mlp4", models.mlp4 + models.outlayer)
exp.add_model("mlp5", models.mlp5 + models.outlayer)
exp.add_model("convnet1", models.convnet1 + models.outlayer)
exp.add_model("convnet2", models.convnet2 + models.outlayer)
exp.add_model("convnet3", models.convnet3 + models.outlayer)
exp.add_model("convnet4", models.convnet4 + models.outlayer)
exp.add_model("convnet5", models.convnet5 + models.outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare models
for tname, iname, lname in [(x, y, z) for x in exp.trainers for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, "mlp*", tname, iname, lname, ".state"),
                        exp.path(trial, "mlp", tname, iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, "convnet*", tname, iname, lname, ".state"),
                        exp.path(trial, "convnet", tname, iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, ".*", tname, iname, lname, ".state"),
                        exp.path(trial, None, tname, iname, lname, ".pdf"))

        exp.summarize(trials, "mlp*", tname, iname, lname,
                exp.path(None, "mlp", tname, iname, lname, ".log"),
                exp.path(None, "mlp", tname, iname, lname, ".csv"))

        exp.summarize(trials, "convnet*", tname, iname, lname,
                exp.path(None, "convnet", tname, iname, lname, ".log"),
                exp.path(None, "convnet", tname, iname, lname, ".csv"))

        exp.summarize(trials, ".*", tname, iname, lname,
                exp.path(None, None, tname, iname, lname, ".log"),
                exp.path(None, None, tname, iname, lname, ".csv"))
