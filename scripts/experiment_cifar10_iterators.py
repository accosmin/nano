import config
import experiment
import models_cifar10 as models

# initialize experiment:
# - single-class classification problem using the CIFAR-10 dataset
# - the model should predict the object of a RGB image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/cifar10/eval_enhancers",
        trials = 10)

# loss functions
exp.add_loss("slogistic")

# enhancers
exp.add_enhancer("default")
exp.add_enhancer("noise", "noise=0.10", "noise10")
exp.add_enhancer("noise", "noise=0.20", "noise20")
exp.add_enhancer("noise", "noise=0.40", "noise40")
exp.add_enhancer("noise", "noise=0.80", "noise80")
exp.add_enhancer("noclass", "ratio=0.10,noise=0.00", "noclass10")
exp.add_enhancer("noclass", "ratio=0.20,noise=0.00", "noclass20")
exp.add_enhancer("noclass", "ratio=0.40,noise=0.00", "noclass40")
exp.add_enhancer("noclass", "ratio=0.80,noise=0.00", "noclass80")
exp.add_enhancer("warp")

# trainers
batch_params = "epochs=1000,patience=32,epsilon=1e-6"
stoch_params = "epochs=1000,patience=32,epsilon=1e-6,batch=32"

exp.add_trainer("stoch_adadelta", stoch_params)

# models
exp.add_model("convnet9", models.convnet9 + models.outlayer)

# train all configurations
exp.run_all()

# compare configurations
exp.summarize_by_enhancers(".*")
