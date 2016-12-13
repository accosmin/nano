import os

class config:
        def __init__(self):
                # useful paths
                homedir = os.path.expanduser('~')
                self.expdir = homedir + "/experiments/results"
                self.dbdir = homedir + "/experiments/databases"

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_train = crtpath + "/../build-release/apps/train"
                self.app_stats = crtpath + "/../build-release/apps/stats"
                self.app_info = crtpath + "/../build-release/apps/info"

                # available losses: {name, command line parameters}+
                self.losses = {
                        "loss_cauchy" : "--loss cauchy",
                        "loss_classnll" : "--loss classnll",
                        "loss_exponential" : "--loss exponential",
                        "loss_logistic" : "--loss logistic",
                        "loss_square" : "--loss square"
                }

                # available criteria: {name, command line parameters}+
                self.criteria = {
                        "crit_avg" : "--criterion avg",
                        "crit_avg_l2n" : "--criterion avg-l2n",
                        "crit_avg_var" : "--criterion avg-var",
                        "crit_max" : "--criterion max",
                        "crit_max_l2n" : "--criterion max-l2n",
                        "crit_max_var" : "--criterion max-var"
                }

                # training policies
                self.policies = {
                        "stop_early" : ",policy=stop_early",
                        "all_epochs" : ",policy=all_epochs"
                }

                # training methods
                self.trainers = {
                        "stoch_ag" : "--trainer stoch --trainer-params opt=ag",
                        "stoch_agfr" : "--trainer stoch --trainer-params opt=agfr",
                        "stoch_aggr" : "--trainer stoch --trainer-params opt=aggr",
                        "stoch_sg" : "--trainer stoch --trainer-params opt=sg",
                        "stoch_sgm" : "--trainer stoch --trainer-params opt=sgm",
                        "stoch_ngd" : "--trainer stoch --trainer-params opt=ngd",
                        "stoch_svrg" : "--trainer stoch --trainer-params opt=svrg",
                        "stoch_asgd" : "--trainer stoch --trainer-params opt=asgd",
                        "stoch_adagrad" : "--trainer stoch --trainer-params opt=adagrad",
                        "stoch_adadelta" : "--trainer stoch --trainer-params opt=adadelta",
                        "stoch_adam" : "--trainer stoch --trainer-params opt=adam",

                        "batch_lbfgs" : "--trainer batch --trainer-params opt=lbfgs",
                        "batch_cgd" : "--trainer batch --trainer-params opt=cgd",
                        "batch_gd" : "--trainer batch --trainer-params opt=gd"
                }
