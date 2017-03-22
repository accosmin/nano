import config
import experiment

# initialize experiment:
# - single-class classification problem using the WINE dataset
# - the model should predict the wine quality class
cfg = config.config()
exp = experiment.experiment(
        cfg.task_wine(),
        cfg.expdir + "/wine/eval_models")

# loss functions
exp.add_losses([
        "loss_classnll"])

# criteria
exp.add_criteria([
        "crit_avg"])

# trainers
exp.add_trainers([
        "batch_cgd"])

# models
outlayer = "affine:dims=3;act-ewave:alpha=1;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-ewave:alpha=1;"
mlp2 = mlp1 + "affine:dims=128;act-ewave:alpha=1;"
mlp3 = mlp2 + "affine:dims=128;act-ewave:alpha=1;"
mlp4 = mlp3 + "affine:dims=128;act-ewave:alpha=1;"

def add_model(name, params, activation):
        name = (name + "-act-" + activation).replace("-", "_").replace(":alpha=", "")
        params = (params + outlayer).replace("act-snorm", "act-" + activation)
        exp.add_model(name, params)

for activation in ["snorm", "tanh", "sin", "ewave:alpha=1", "ewave:alpha=2", "ewave:alpha=3", "ewave:alpha=4", "pwave"]:
        add_model("mlp0", mlp0, activation)
        add_model("mlp1", mlp1, activation)
        add_model("mlp2", mlp2, activation)
        add_model("mlp3", mlp3, activation)
        add_model("mlp4", mlp4, activation)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 1000, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for tname in exp.trainers:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                # compare activation functions for each model
                                exp.plot_many(
                                        exp.filter(trial, "mlp0.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp0", tname, cname, lname, ".pdf"))
                                exp.plot_many(
                                        exp.filter(trial, "mlp1.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp1", tname, cname, lname, ".pdf"))
                                exp.plot_many(
                                        exp.filter(trial, "mlp2.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp2", tname, cname, lname, ".pdf"))
                                exp.plot_many(
                                        exp.filter(trial, "mlp3.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp3", tname, cname, lname, ".pdf"))
                                exp.plot_many(
                                        exp.filter(trial, "mlp4.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp4", tname, cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
