from config import *
from experiment import *

# initialize experiment:
# - classification problem: predict the parity bit of binary inputs
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_nparity", trials = 10)

exp.set_task(cfg.task_synth_nparity(n = 8, count = 10000))

# loss functions
exp.add_loss("logistic", cfg.loss("s-logistic"))

# trainers
epochs = 100
patience = 100
epsilon = 1e-6

for solver in cfg.batch_solvers():
        exp.add_trainer("batch_{}".format(solver), cfg.batch_trainer(solver, epochs, patience, epsilon))

for solver in cfg.stoch_solvers():
        exp.add_trainer("stoch_{}".format(solver), cfg.stoch_trainer(solver, epochs, patience, epsilon))

# models
output = {"name":"output","type":"affine","config":{"omaps":1,"orows":1,"ocols":1}}

fc1 = {"name":"fc1","type":"affine","config":{"omaps":16,"orows":1,"ocols":1}}
fc2 = {"name":"fc2","type":"affine","config":{"omaps":32,"orows":1,"ocols":1}}
fc3 = {"name":"fc3","type":"affine","config":{"omaps":64,"orows":1,"ocols":1}}

ac1 = {"name":"ac1","type":"act-snorm","config":{}}
ac2 = {"name":"ac2","type":"act-snorm","config":{}}
ac3 = {"name":"ac3","type":"act-snorm","config":{}}

mlp0 = {"nodes": [output], "model": []}
mlp1 = {"nodes": [fc1, ac1, output], "model": [["fc1", "ac1", "output"]]}
mlp2 = {"nodes": [fc1, ac1, fc2, ac2, output], "model": [["fc1", "ac1", "fc2", "ac2", "output"]]}
mlp3 = {"nodes": [fc1, ac1, fc2, ac2, fc3, ac3, output], "model": [["fc1", "ac1", "fc2", "ac2", "fc3", "ac3", "output"]]}

exp.add_model("mlp0", mlp0)
exp.add_model("mlp1", mlp1)
exp.add_model("mlp2", mlp2)
exp.add_model("mlp3", mlp3)

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "stoch_*")
exp.summarize_by_trainers("batch", "batch_*")
exp.summarize_by_trainers("all", ".*")

exp.summarize_by_models("all", ".*")
