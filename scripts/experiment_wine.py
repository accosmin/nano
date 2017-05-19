import config
import experiment

# initialize experiment:
# - single-class classification problem using the WINE dataset
# - the model should predict the wine quality class
cfg = config.config()
exp = experiment.experiment(
        cfg.task_wine(),
        cfg.expdir + "/wine")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
exp.add_trainer("batch_cgd", "epochs=1000,policy=stop_early,patience=100")

# models
outlayer = "affine:dims=3;act-snorm;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-snorm;"
mlp2 = mlp1 + "affine:dims=128;act-snorm;"
mlp3 = mlp2 + "affine:dims=128;act-snorm;"
mlp4 = mlp3 + "affine:dims=128;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)
exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("mlp3", mlp3 + outlayer)
exp.add_model("mlp4", mlp4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials)

# compare models
for tname, iname, lname in [(x, y, z) for x in exp.trainers for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, ".*", tname, iname, lname, ".state"),
                        exp.path(trial, None, tname, iname, lname, ".pdf"))

        exp.summarize(trials, ".*", tname, iname, lname,
                exp.path(None, None, tname, iname, lname, ".log"),
                exp.path(None, None, tname, iname, lname, ".csv"))
