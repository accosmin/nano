import config
import experiment

# initialize experiment:
# - single-class classification problem using the MNIST dataset
# - the model should predict the digit of a grayscale image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_mnist(),
        cfg.expdir + "/mnist/eval_trainers/")

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=100,patience=32,epsilon=1e-6"
stoch_params = "epochs=100,patience=32,epsilon=1e-6,min_batch=32,max_batch=256"

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
outlayer = "affine:dims=10;"

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + "conv:dims=32,rows=7,cols=7,conn=1,drow=2,dcol=2;act-snorm;"
convnet2 = convnet1 + "conv:dims=32,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet3 = convnet2 + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"
convnet4 = convnet3 + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"

exp.add_model("convnet4", convnet4 + outlayer)

# train all configurations
trials = 10
exp.run_all(trials = trials)

# compare trainers
for mname, iname, lname in [(x, y, z) for x in exp.models for y in exp.iterators for z in exp.losses]:
        for trial in range(trials):
                exp.plot_many(
                        exp.filter(trial, mname, "stoch*", iname, lname, ".state"),
                        exp.path(trial, mname, "stoch", iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, mname, "batch*", iname, lname, ".state"),
                        exp.path(trial, mname, "batch", iname, lname, ".pdf"))

                exp.plot_many(
                        exp.filter(trial, mname, ".*", iname, lname, ".state"),
                        exp.path(trial, mname, None, iname, lname, ".pdf"))

        exp.summarize(trials, mname, "stoch*", iname, lname,
                exp.path(None, mname, "stoch", iname, lname, ".log"),
                exp.path(None, mname, "stoch", iname, lname, ".csv"))

        exp.summarize(trials, mname, "batch*", iname, lname,
                exp.path(None, mname, "batch", iname, lname, ".log"),
                exp.path(None, mname, "batch", iname, lname, ".csv"))

        exp.summarize(trials, mname, ".*", iname, lname,
                exp.path(None, mname, None, iname, lname, ".log"),
                exp.path(None, mname, None, iname, lname, ".csv"))
