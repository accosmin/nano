from config import *
from experiment import *

# initialize experiment:
# - regression problem: predict the position of a peak in an image
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_peak2d", trials = 10)

exp.set_task(cfg.task_synth_peak2d(irows = 16, icols = 16, noise = 0.0, count = 4000))

# loss functions
exp.add_loss("cauchy", cfg.loss("cauchy"))

# trainers
epochs = 100
patience = 100
epsilon = 1e-6

for solver in cfg.batch_solvers():
        exp.add_trainer("batch_{}".format(solver), cfg.batch_trainer(solver, epochs, patience, epsilon))

for solver in cfg.stoch_solvers():
        exp.add_trainer("stoch_{}".format(solver), cfg.stoch_trainer(solver, epochs, patience, epsilon))

# models
output = {"name":"output","type":"affine","config":{"omaps":2,"orows":1,"ocols":1}}

cn1 = {"name":"cn1","type":"conv2d","config":{"omaps":16,"krows":5,"kcols":5,"kconn":1,"kdrow":1,"kdcol":1}}
cn2 = {"name":"cn2","type":"conv2d","config":{"omaps":32,"krows":5,"kcols":5,"kconn":1,"kdrow":1,"kdcol":1}}
cn3 = {"name":"cn3","type":"conv2d","config":{"omaps":64,"krows":5,"kcols":5,"kconn":1,"kdrow":1,"kdcol":1}}

ac1 = {"name":"ac1","type":"act-snorm","config":{}}
ac2 = {"name":"ac2","type":"act-snorm","config":{}}
ac3 = {"name":"ac3","type":"act-snorm","config":{}}

cnn1 = {"nodes": [cn1, ac1, output], "model": ["cn1", "ac1", "output"]}
cnn2 = {"nodes": [cn1, ac1, cn2, ac2, output], "model": ["cn1", "ac1", "cn2", "ac2", "output"]}
cnn3 = {"nodes": [cn1, ac1, cn2, ac2, cn3, ac3, output], "model": ["cn1", "ac1", "cn2", "ac2", "cn3", "ac3", "output"]}

exp.add_model("cnn1", cnn1)
exp.add_model("cnn2", cnn2)
exp.add_model("cnn3", cnn3)

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "stoch_*")
exp.summarize_by_trainers("batch", "batch_*")
exp.summarize_by_trainers("all", ".*")

exp.summarize_by_models("all", ".*")
