import json
from config import *
from experiment import *

# initialize experiment:
# - regression problem: predict the output of an affine transformation
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_affine", trials = 10)

exp.set_task(cfg.task_synth_affine(isize = 16, osize = 8, noise = 0.1, count = 10000))

# loss functions
exp.add_loss("cauchy", cfg.loss("cauchy"))

# trainers
epochs = 100
patience = 100
epsilon = 1e-4

for solver in cfg.batch_solvers():
        exp.add_trainer(solver, cfg.batch_trainer(solver, epochs, patience, epsilon))

for solver in cfg.stoch_solvers():
        exp.add_trainer(solver, cfg.stoch_trainer(solver, epochs, patience, epsilon))

# models
output = {"name":"output","type":"affine","omaps":8,"orows":1,"ocols":1}

mlp0 = {"nodes": [output], "model": []}

exp.add_model("mlp0", mlp0)

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "|".join(cfg.stoch_solvers()))
exp.summarize_by_trainers("batch", "|".join(cfg.batch_solvers()))
exp.summarize_by_trainers("all", ".*")
