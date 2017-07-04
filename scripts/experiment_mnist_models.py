import config
import experiment
import models_mnist as models

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_models/")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("mlp0", models.mlp0 + models.outlayer)
exp.add_model("mlp1", models.mlp1 + models.outlayer)
exp.add_model("mlp2", models.mlp2 + models.outlayer)
exp.add_model("mlp3", models.mlp3 + models.outlayer)
exp.add_model("mlp4", models.mlp4 + models.outlayer)
exp.add_model("mlp5", models.mlp5 + models.outlayer)
exp.add_model("convnet1", models.convnet1 + models.outlayer)
exp.add_model("convnet2", models.convnet2 + models.outlayer)
exp.add_model("convnet3", models.convnet3 + models.outlayer)
exp.add_model("convnet4", models.convnet4 + models.outlayer)
exp.add_model("convnet5", models.convnet5 + models.outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare models
exp.summarize_by_models(trials = trials)
