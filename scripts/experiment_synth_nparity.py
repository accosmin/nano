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

exp.add_trainer("gd", cfg.batch_trainer("gd", epochs, patience, epsilon))
exp.add_trainer("cgd", cfg.batch_trainer("cgd", epochs, patience, epsilon))
exp.add_trainer("lbfgs", cfg.batch_trainer("lbfgs", epochs, patience, epsilon))

exp.add_trainer("ag", cfg.stoch_trainer("ag", epochs, patience, epsilon))
exp.add_trainer("agfr", cfg.stoch_trainer("agfr", epochs, patience, epsilon))
exp.add_trainer("aggr", cfg.stoch_trainer("aggr", epochs, patience, epsilon))

exp.add_trainer("sg", cfg.stoch_trainer("sg", epochs, patience, epsilon))
exp.add_trainer("sgm", cfg.stoch_trainer("sgm", epochs, patience, epsilon))
exp.add_trainer("ngd", cfg.stoch_trainer("ngd", epochs, patience, epsilon))
exp.add_trainer("asgd", cfg.stoch_trainer("asgd", epochs, patience, epsilon))
exp.add_trainer("svrg", cfg.stoch_trainer("svrg", epochs, patience, epsilon))
exp.add_trainer("rmsprop", cfg.stoch_trainer("rmsprop", epochs, patience, epsilon))

exp.add_trainer("adam", cfg.stoch_trainer("adam", epochs, patience, epsilon))
exp.add_trainer("adagrad", cfg.stoch_trainer("adagrad", epochs, patience, epsilon))
exp.add_trainer("amsgrad", cfg.stoch_trainer("amsgrad", epochs, patience, epsilon))
exp.add_trainer("adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon))
exp.add_trainer("cocob", cfg.stoch_trainer("cocob", epochs, patience, epsilon))

# models
output = {"name":"output","type":"affine","config":{"omaps":1,"orows":1,"ocols":1}}

fc1 = {"name":"fc1","type":"affine","config":{"omaps":16,"orows":1,"ocols":1}}
fc2 = {"name":"fc2","type":"affine","config":{"omaps":32,"orows":1,"ocols":1}}
fc3 = {"name":"fc3","type":"affine","config":{"omaps":64,"orows":1,"ocols":1}}

ac1 = {"name":"ac1","type":"act-snorm","config":{}}
ac2 = {"name":"ac2","type":"act-snorm","config":{}}
ac3 = {"name":"ac3","type":"act-snorm","config":{}}

mlp0 = {"nodes": [output], "model": []}
mlp1 = {"nodes": [fc1, ac1, output], "model": ["fc1", "ac1", "output"]}
mlp2 = {"nodes": [fc1, ac1, fc2, ac2, output], "model": ["fc1", "ac1", "fc2", "ac2", "output"]}
mlp3 = {"nodes": [fc1, ac1, fc2, ac2, fc3, ac3, output], "model": ["fc1", "ac1", "fc2", "ac2", "fc3", "ac3", "output"]}

exp.add_model("mlp0", mlp0)
exp.add_model("mlp1", mlp1)
exp.add_model("mlp2", mlp2)
exp.add_model("mlp3", mlp3)

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "ag|agfr|aggr|sg|sgm|ngd|asgd|svrg|rmsprop|adam|adagrad|amsgrad|adadelta|cocob")
exp.summarize_by_trainers("batch", "gd|cgd|lbfgs")
exp.summarize_by_trainers("all", ".*")

exp.summarize_by_models("all", ".*")
