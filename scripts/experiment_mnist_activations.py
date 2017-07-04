import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_activations/")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
for activation in ["snorm", "tanh", "sin", "pwave"]:
        name = ("convnet5-act-" + activation).replace("-", "_")
        params = (models.convnet5 + models.outlayer).replace("act-snorm", "act-" + activation)
        exp.add_model(name, params)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare models
exp.summarize_by_models(trials = trials)
