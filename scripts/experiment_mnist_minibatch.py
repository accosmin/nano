import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_minibatch/",
        trials = 10)

# loss functions
exp.add_loss("slogistic")

# iterators
exp.add_iterator("default")

# trainers
stoch_params = "epochs=10,patience=32,epsilon=1e-6,batch={}"
minibatch_name = "batch{}"

for size in [32, 64, 128, 256, 512, 1024]:
        exp.add_trainer("stoch_adadelta", stoch_params.format(size), minibatch_name.format(size))

# models
exp.add_model("mlp0", models.mlp0 + models.outlayer)

# train all configurations
#exp.train_all()

# compare trainers
exp.summarize_by_trainers(".*")
