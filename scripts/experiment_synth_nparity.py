from config import *
from experiment import *

# initialize experiment:
# - classification problem: predict the parity bit of binary inputs
cfg = config.config()
exp = experiment(cfg.expdir + "/synth_nparity", trials = 10)

exp.set_task(cfg.task_synth_nparity(n = 10, count = 10000))

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
exp.add_trainer("adadelta", cfg.stoch_trainer("adadelta", epochs, patience, epsilon))

# models
mlp0 = []
mlp1 = mlp0 + [20,1,1]
mlp2 = mlp1 + [30,1,1]
mlp3 = mlp2 + [40,1,1]
mlp4 = mlp3 + [50,1,1]

exp.add_model("mlp0", cfg.mlp(mlp0, imaps=10, irows=1, icols=1, omaps=1, orows=1, ocols=1))
exp.add_model("mlp1", cfg.mlp(mlp1, imaps=10, irows=1, icols=1, omaps=1, orows=1, ocols=1))
exp.add_model("mlp2", cfg.mlp(mlp2, imaps=10, irows=1, icols=1, omaps=1, orows=1, ocols=1))
exp.add_model("mlp3", cfg.mlp(mlp3, imaps=10, irows=1, icols=1, omaps=1, orows=1, ocols=1))
exp.add_model("mlp4", cfg.mlp(mlp4, imaps=10, irows=1, icols=1, omaps=1, orows=1, ocols=1))

# train all configurations
exp.train_all()

# compare configurations
exp.summarize_by_trainers("stoch", "ag|agfr|aggr|sg|sgm|ngd|asgd|svrg|rmsprop|adam|adagrad|adadelta")
exp.summarize_by_trainers("batch", "gd|cgd|lbfgs")
exp.summarize_by_trainers("all", ".*")
