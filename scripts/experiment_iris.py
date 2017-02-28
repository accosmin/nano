import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
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
        "stoch_svrg"])

# models
outlayer = "affine:dims=3;act-snorm;"

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=16;act-snorm;"
mlp2 = mlp1 + "affine:dims=16;act-snorm;"
mlp3 = mlp2 + "affine:dims=16;act-snorm;"
mlp4 = mlp3 + "affine:dims=16;act-snorm;"

exp.add_model("mlp0", mlp0 + outlayer)
exp.add_model("mlp1", mlp1 + outlayer)
exp.add_model("mlp2", mlp2 + outlayer)
exp.add_model("mlp3", mlp3 + outlayer)
exp.add_model("mlp4", mlp4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials, epochs = 100, policy = "stop_early", min_batch = 32, max_batch = 32)

# compare configurations
for trial in range(trials):
        for cname in exp.criteria:
                for lname in exp.losses:
                        break
                        # compare mlps
                        exp.plot_many(
                                exp.filter(trial, "mlp*", ".*", cname, lname, ".state"),
                                exp.get_path(trial, "mlp", "", cname, lname, ".pdf"))

# summarize configurations
exp.summarize(trials)
