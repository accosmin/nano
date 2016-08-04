#include "nano.h"

#include "losses/square.h"
#include "losses/cauchy.h"
#include "losses/logistic.h"
#include "losses/classnll.h"
#include "losses/exponential.h"

#include "tasks/task_mnist.h"
#include "tasks/task_cifar10.h"
#include "tasks/task_cifar100.h"
#include "tasks/task_stl10.h"
#include "tasks/task_svhn.h"
#include "tasks/task_charset.h"
#include "tasks/task_affine.h"

#include "layers/activation_unit.h"
#include "layers/activation_tanh.h"
#include "layers/activation_snorm.h"
#include "layers/activation_splus.h"
#include "layers/convolution_kernel2d.h"
#include "layers/convolution_toeplitz.h"
#include "layers/affine.h"
#include "layers/pooling_full.h"
#include "layers/pooling_soft.h"
#include "layers/pooling_gauss.h"

#include "models/forward_network.h"

#include "trainers/batch.h"
#include "trainers/stochastic.h"

#include "criteria/l2nreg.h"
#include "criteria/varreg.h"

#include "optim/stoch/ag.h"
#include "optim/stoch/adam.h"
#include "optim/stoch/adagrad.h"
#include "optim/stoch/adadelta.h"
#include "optim/stoch/ngd.h"
#include "optim/stoch/sg.h"
#include "optim/stoch/sgm.h"

#include "optim/batch/gd.h"
#include "optim/batch/cgd.h"
#include "optim/batch/lbfgs.h"

#include <cfenv>

namespace nano
{
        task_manager_t& get_tasks()
        {
                static task_manager_t manager;
                return manager;
        }

        layer_manager_t& get_layers()
        {
                static layer_manager_t manager;
                return manager;
        }

        model_manager_t& get_models()
        {
                static model_manager_t manager;
                return manager;
        }

        loss_manager_t& get_losses()
        {
                static loss_manager_t manager;
                return manager;
        }

        criterion_manager_t& get_criteria()
        {
                static criterion_manager_t manager;
                return manager;
        }

        trainer_manager_t& get_trainers()
        {
                static trainer_manager_t manager;
                return manager;
        }

        stoch_optimizer_manager_t& get_stoch_optimizers()
        {
                static stoch_optimizer_manager_t manager;
                return manager;
        }

        batch_optimizer_manager_t& get_batch_optimizers()
        {
                static batch_optimizer_manager_t manager;
                return manager;
        }

        struct init_t
        {
                init_t()
                {
                        // round to nearest integer
                        std::fesetround(FE_TONEAREST);

                        // use Eigen with multiple threads
                        Eigen::initParallel();
                        Eigen::setNbThreads(0);

                        // register losses
                        nano::get_losses().add("square", square_loss_t());
                        nano::get_losses().add("cauchy", cauchy_loss_t());
                        nano::get_losses().add("logistic", logistic_loss_t());
                        nano::get_losses().add("classnll", classnll_loss_t());
                        nano::get_losses().add("exponential", exponential_loss_t());

                        // register tasks
                        nano::get_tasks().add("mnist", mnist_task_t());
                        nano::get_tasks().add("cifar10", cifar10_task_t());
                        nano::get_tasks().add("cifar100", cifar100_task_t());
                        nano::get_tasks().add("stl10", stl10_task_t());
                        nano::get_tasks().add("svhn", svhn_task_t());
                        nano::get_tasks().add("charset", charset_task_t());
                        nano::get_tasks().add("affine", affine_task_t());

                        // register layers
                        nano::get_layers().add("act-unit", unit_activation_layer_t());
                        nano::get_layers().add("act-tanh", tanh_activation_layer_t());
                        nano::get_layers().add("act-snorm", snorm_activation_layer_t());
                        nano::get_layers().add("act-splus", softplus_activation_layer_t());
                        nano::get_layers().add("affine", affine_layer_t());
                        nano::get_layers().add("conv-k2d", conv_layer_kernel2d_t());
                        nano::get_layers().add("conv-toe", conv_layer_toeplitz_t());
                        nano::get_layers().add("conv", conv_layer_toeplitz_t());
                        nano::get_layers().add("pool-full", pooling_full_layer_t());
                        nano::get_layers().add("pool-soft", pooling_soft_layer_t());
                        nano::get_layers().add("pool-gauss", pooling_gauss_layer_t());

                        // register models
                        nano::get_models().add("forward-network", forward_network_t());

                        // register trainers
                        nano::get_trainers().add("batch", batch_trainer_t());
                        nano::get_trainers().add("stochastic", stochastic_trainer_t());

                        // register criteria
                        nano::get_criteria().add("avg", average_criterion_t());
                        nano::get_criteria().add("avg-l2n", average_l2n_criterion_t());
                        nano::get_criteria().add("avg-var", average_var_criterion_t());
                        nano::get_criteria().add("max", softmax_criterion_t());
                        nano::get_criteria().add("max-l2n", softmax_l2n_criterion_t());
                        nano::get_criteria().add("max-var", softmax_var_criterion_t());

                        // register stochastic optimizers
                        nano::get_stoch_optimizers().add("sg", stoch_sg_t());
                        nano::get_stoch_optimizers().add("sgm", stoch_sgm_t());
                        nano::get_stoch_optimizers().add("ngd", stoch_ngd_t());
                        nano::get_stoch_optimizers().add("ag", stoch_ag_t());
                        nano::get_stoch_optimizers().add("agfr", stoch_agfr_t());
                        nano::get_stoch_optimizers().add("aggr", stoch_aggr_t());
                        nano::get_stoch_optimizers().add("adam", stoch_adam_t());
                        nano::get_stoch_optimizers().add("adagrad", stoch_adagrad_t());
                        nano::get_stoch_optimizers().add("adadelta", stoch_adadelta_t());

                        // register batch optimizers
                        nano::get_batch_optimizers().add("gd", batch_gd_t());
                        nano::get_batch_optimizers().add("cgd", batch_cgd_prp_t());
                        nano::get_batch_optimizers().add("cgd-n", batch_cgd_n_t());
                        nano::get_batch_optimizers().add("cgd-hs", batch_cgd_hs_t());
                        nano::get_batch_optimizers().add("cgd-fr", batch_cgd_fr_t());
                        nano::get_batch_optimizers().add("cgd-prp", batch_cgd_prp_t());
                        nano::get_batch_optimizers().add("cgd-cd", batch_cgd_cd_t());
                        nano::get_batch_optimizers().add("cgd-ls", batch_cgd_ls_t());
                        nano::get_batch_optimizers().add("cgd-dy", batch_cgd_dy_t());
                        nano::get_batch_optimizers().add("cgd-dycd", batch_cgd_dycd_t());
                        nano::get_batch_optimizers().add("cgd-dyhs", batch_cgd_dyhs_t());
                        nano::get_batch_optimizers().add("lbfgs", batch_lbfgs_t());
                }
        };

        static const init_t the_initializer;
}

