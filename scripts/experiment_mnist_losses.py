import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_losses",
        trials = 10)

# loss functions
exp.add_loss("classnll")
exp.add_loss("slogistic")
exp.add_loss("sexponential")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("convnet5", models.convnet5 + models.outlayer)

# train all configurations
exp.run_all()

# compare configurations
exp.summarize_by_losses(".*")
