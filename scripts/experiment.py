import os

class config:
        def __init__(self):
                # useful paths
                homedir = os.path.expanduser('~')
                self.expdir = homedir + "/experiments/results"
                self.dbdir = homedir + "/experiments/databases"

                crtpath = os.path.dirname(os.path.realpath(__file__))
                self.app_trainer = crtpath + "/../build-release/apps/trainer"
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
                        stoch_ag="--trainer stoch --trainer-params opt=ag,epochs=${epochs},policy=${policy}"
                        stoch_agfr="--trainer stoch --trainer-params opt=agfr,epochs=${epochs},policy=${policy}"
                        stoch_aggr="--trainer stoch --trainer-params opt=aggr,epochs=${epochs},policy=${policy}"
                        stoch_sg="--trainer stoch --trainer-params opt=sg,epochs=${epochs},policy=${policy}"
                        stoch_sgm="--trainer stoch --trainer-params opt=sgm,epochs=${epochs},policy=${policy}"
                        stoch_ngd="--trainer stoch --trainer-params opt=ngd,epochs=${epochs},policy=${policy}"
                        stoch_svrg="--trainer stoch --trainer-params opt=svrg,epochs=${epochs},policy=${policy}"
                        stoch_asgd="--trainer stoch --trainer-params opt=asgd,epochs=${epochs},policy=${policy}"
                        stoch_adagrad="--trainer stoch --trainer-params opt=adagrad,epochs=${epochs},policy=${policy}"
                        stoch_adadelta="--trainer stoch --trainer-params opt=adadelta,epochs=${epochs},policy=${policy}"
                        stoch_adam="--trainer stoch --trainer-params opt=adam,epochs=${epochs},policy=${policy}"

                        batch_lbfgs="--trainer batch --trainer-params opt=lbfgs,epochs=${epochs},policy=${policy}"
                        batch_cgd="--trainer batch --trainer-params opt=cgd,epochs=${epochs},policy=${policy}"
                        batch_gd="--trainer batch --trainer-params opt=gd,epochs=${epochs},policy=${policy}"


class experiment:
        def __init__(self, task_params, outdir):
                self.task_params = task_params
                self.outdir = outdir
                self.models = {}
                self.trainers = {}
                self.criteria = {}
                self.losses = {}

        def add_model(self, name, params):
                self.models[name] = params

        def add_trainer(self, name, params):
                self.trainers[name] = params

        def add_loss(self, name, params):
                self.losses[name] = params

        def add_criterion(self, name, params):
                self.criteria[name] = params

        def run(self, mname, mparams, tname, tparams, cname, cparams, lname, lparams):
                print("model: ", mname, "trainer:", tname, "criterion:", cname, "loss:", lname)
                print("todo")

        def run(self, trials, epochs):
                os.mkdirs(self.output_dir, exist_ok = True)
                for mname, mparams in self.models:
                        for tname, tparams in self.trainers:
                                for cname, cparams in self.criteria:
                                        for lname, lparams in self.losses:
                                                self.run(mname, mparams, tname, tparams, cname, cparams, lname, lparams)

