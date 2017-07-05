import config
import experiment

# initialize experiment:
# - single-class classification problem using a synthetic task
# - the model should predict the digit of a synthetic image
cfg = config.config()
exp = experiment.experiment(
        cfg.task_synth_charset(ctype = "digit", color = "rgb", irows = 16, icols = 16, count = 10000),
        cfg.expdir + "/charset",
        trials = 10)

# loss functions
exp.add_loss("classnll")

# iterators
exp.add_iterator("default")

# trainers
batch_params = "epochs=100,patience=32,epsilon=1e-6"
stoch_params = "epochs=100,patience=32,epsilon=1e-6,batch=32"

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
mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + "affine:dims=128;act-snorm;"
mlp2 = mlp1 + "affine:dims=64;act-snorm;"
mlp3 = mlp2 + "affine:dims=32;act-snorm;"
mlp4 = mlp3 + "affine:dims=32;act-snorm;"
mlp5 = mlp4 + "affine:dims=32;act-snorm;"

convnet = "--model forward-network --model-params "
convnet = convnet + "conv:dims=64,rows=5,cols=5,conn=1,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=32,rows=3,cols=3,conn=1,drow=1,dcol=1;act-snorm;"
convnet = convnet + "conv:dims=32,rows=1,cols=1,conn=1,drow=1,dcol=1;act-snorm;"

outlayer = "affine:dims=10;"

#exp.add_model("mlp0", mlp0 + outlayer)
#exp.add_model("mlp1", mlp1 + outlayer)
#exp.add_model("mlp2", mlp2 + outlayer)
#exp.add_model("mlp3", mlp3 + outlayer)
#exp.add_model("mlp4", mlp4 + outlayer)
#exp.add_model("mlp5", mlp5 + outlayer)
exp.add_model("convnet", convnet + outlayer)

# train all configurations
exp.run_all()

# compare configurations
exp.summarize_by_models(".*")
exp.summarize_by_trainers("stoch*")
exp.summarize_by_trainers("batch*")
exp.summarize_by_trainers(".*")
