import config
import experiment

# initialize experiment:
# - single-label classification problem using the IRIS flower dataset
# - the model should predict the iris species
cfg = config.config()
exp = experiment.experiment(cfg.expdir + "/iris", trials = 10)

exp.set_task(cfg.task_iris(folds=10))

# loss functions
exp.add_loss("cauchy", cfg.loss("s-cauchy"))
exp.add_loss("square", cfg.loss("s-square"))
exp.add_loss("classnll", cfg.loss("classnll"))
exp.add_loss("logistic", cfg.loss("s-logistic"))
exp.add_loss("exponential", cfg.loss("s-exponential"))

# models
gboost_stump_real = {"id": "gboost-stump", "rounds": 100, "patience": 10, "solver": "gd", "type": "real" }
gboost_stump_discrete = {"id": "gboost-stump", "rounds": 100, "patience": 10, "solver": "gd", "type": "discrete" }

exp.add_model("gboost-stump-real", gboost_stump_real)
exp.add_model("gboost-stump-discrete", gboost_stump_discrete)

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_losses("all", ".*")
exp.summarize_by_models("all", ".*")
