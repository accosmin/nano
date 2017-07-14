import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_trainers/",
        trials = 10)

# loss functions
exp.add_loss("classnll")

# enhancers
exp.add_enhancer("default")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("batch_gd", batch_params)
exp.add_trainer("batch_cgd", batch_params)
exp.add_trainer("batch_lbfgs", batch_params)

exp.add_trainer("stoch_ag", stoch_params)
exp.add_trainer("stoch_agfr", stoch_params)
exp.add_trainer("stoch_aggr", stoch_params)

exp.add_trainer("stoch_sg", stoch_params)
exp.add_trainer("stoch_sgm", stoch_params)
exp.add_trainer("stoch_ngd", stoch_params)
exp.add_trainer("stoch_asgd", stoch_params)
exp.add_trainer("stoch_svrg", stoch_params)
exp.add_trainer("stoch_rmsprop", stoch_params)

exp.add_trainer("stoch_adam", stoch_params)
exp.add_trainer("stoch_adagrad", stoch_params)
exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("convnet5", models.convnet5 + models.outlayer)

# train all configurations
exp.run_all()

# compare trainers
exp.summarize_by_trainers(".*")
