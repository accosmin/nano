import config
import experiment

# initialize experiment:
# - single-class classification problem using the IRIS flower dataset
# - the model should predict the iris species
cfg = config.config()
exp = experiment.experiment(
        cfg.task_iris(),
        cfg.expdir + "/iris/eval_models")

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
outlayer = "affine:dims=3;act-pwave;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-pwave;"
mlp2 = mlp1 + "affine:dims=128;act-pwave;"
mlp3 = mlp2 + "affine:dims=128;act-pwave;"
mlp4 = mlp3 + "affine:dims=128;act-pwave;"

exp.add_model("mlp0", mlp0 + outlayer)
exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("mlp3", mlp3 + outlayer)
exp.add_model("mlp4", mlp4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 1000, policy = "stop_early")

# compare configurations
for trial in range(trials):
        for tname in exp.trainers:
                for cname in exp.criteria:
                        for lname in exp.losses:
                                # compare mlps
                                exp.plot_many(
                                        exp.filter(trial, "mlp.*", tname, cname, lname, ".state"),
                                        exp.get_path(trial, "mlp", tname, cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
