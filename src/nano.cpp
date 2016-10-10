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
#include "layers/convolution.h"
#include "layers/affine.h"

#include "models/forward_network.h"

#include "trainers/batch.h"
#include "trainers/stochastic.h"

#include "criteria/l2nreg.h"
#include "criteria/varreg.h"

#include "stoch/ag.h"
#include "stoch/adam.h"
#include "stoch/adagrad.h"
#include "stoch/adadelta.h"
#include "stoch/ngd.h"
#include "stoch/sg.h"
#include "stoch/sgm.h"

#include "batch/gd.h"
#include "batch/cgd.h"
#include "batch/lbfgs.h"

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

        static void init_batch_optimizers()
        {
                nano::get_batch_optimizers().add("gd", "gradient descent", batch_gd_t());
                nano::get_batch_optimizers().add("cgd", "nonlinear conjugate gradient descent (default)", batch_cgd_prp_t());
                nano::get_batch_optimizers().add("cgd-n", "nonlinear conjugate gradient descent (N)", batch_cgd_n_t());
                nano::get_batch_optimizers().add("cgd-hs", "nonlinear conjugate gradient descent (HS)", batch_cgd_hs_t());
                nano::get_batch_optimizers().add("cgd-fr", "nonlinear conjugate gradient descent (FR)", batch_cgd_fr_t());
                nano::get_batch_optimizers().add("cgd-prp", "nonlinear conjugate gradient descent (PRP+)", batch_cgd_prp_t());
                nano::get_batch_optimizers().add("cgd-cd", "nonlinear conjugate gradient descent (CD)", batch_cgd_cd_t());
                nano::get_batch_optimizers().add("cgd-ls", "nonlinear conjugate gradient descent (LS)", batch_cgd_ls_t());
                nano::get_batch_optimizers().add("cgd-dy", "nonlinear conjugate gradient descent (DY)", batch_cgd_dy_t());
                nano::get_batch_optimizers().add("cgd-dycd", "nonlinear conjugate gradient descent (DYCD)", batch_cgd_dycd_t());
                nano::get_batch_optimizers().add("cgd-dyhs", "nonlinear conjugate gradient descent (DYHS)", batch_cgd_dyhs_t());
                nano::get_batch_optimizers().add("lbfgs", "limited-memory BFGS", batch_lbfgs_t());
        }

        static void init_stoch_optimizers()
        {
                nano::get_stoch_optimizers().add("sg", "stochastic gradient (descent)", stoch_sg_t());
                nano::get_stoch_optimizers().add("sgm", "stochastic gradient (descent) with momentum", stoch_sgm_t());
                nano::get_stoch_optimizers().add("ngd", "stochastic normalized gradient descent", stoch_ngd_t());
                nano::get_stoch_optimizers().add("ag", "Nesterov's accelerated gradient", stoch_ag_t());
                nano::get_stoch_optimizers().add("agfr", "Nesterov's accelerated gradient with function value restarts", stoch_agfr_t());
                nano::get_stoch_optimizers().add("aggr", "Nesterov's accelerated gradient with gradient restarts", stoch_aggr_t());
                nano::get_stoch_optimizers().add("adam", "Adam (see citation)", stoch_adam_t());
                nano::get_stoch_optimizers().add("adagrad", "AdaGrad (see citation)", stoch_adagrad_t());
                nano::get_stoch_optimizers().add("adadelta", "AdaDelta (see citation)", stoch_adadelta_t());
        }

        static void init_trainers()
        {
                nano::get_trainers().add("batch", "batch trainer", batch_trainer_t());
                nano::get_trainers().add("stoch", "stochastic trainer", stochastic_trainer_t());
        }

        static void init_criteria()
        {
                nano::get_criteria().add("avg", "average loss", average_criterion_t());
                nano::get_criteria().add("avg-l2n", "L2-norm regularized average loss", average_l2n_criterion_t());
                nano::get_criteria().add("avg-var", "variance-regularized average loss", average_var_criterion_t());
                nano::get_criteria().add("max", "softmax loss", softmax_criterion_t());
                nano::get_criteria().add("max-l2n", "L2-norm regularized softmax loss", softmax_l2n_criterion_t());
                nano::get_criteria().add("max-var", "variance-regularized softmax loss", softmax_var_criterion_t());
        }

        static void init_losses()
        {
                nano::get_losses().add("square", "square loss (regression)", square_loss_t());
                nano::get_losses().add("cauchy", "Cauchy loss (regression)", cauchy_loss_t());
                nano::get_losses().add("logistic", "logistic loss (multi-class classification)", logistic_loss_t());
                nano::get_losses().add("classnll", "negative log-likelihood loss (multi-class classification)", classnll_loss_t());
                nano::get_losses().add("exponential", "exponential loss (multi-class classification)", exponential_loss_t());
        }

        static void init_tasks()
        {
                nano::get_tasks().add("mnist", "MNIST (1x28x28 digit classification)", mnist_task_t());
                nano::get_tasks().add("cifar10", "CIFAR-10 (3x32x32 object classification)", cifar10_task_t());
                nano::get_tasks().add("cifar100", "CIFAR-100 (3x32x32 object classification)", cifar100_task_t());
                nano::get_tasks().add("stl10", "STL-10 (3x96x96 semi-supervised object classification)", stl10_task_t());
                nano::get_tasks().add("svhn", "SVHN (3x32x32 digit classification in the wild)", svhn_task_t());
                nano::get_tasks().add("charset", "synthetic character classification", charset_task_t());
                nano::get_tasks().add("affine", "synthetic affine regression", affine_task_t());
        }

        static void init_layers()
        {
                nano::get_layers().add("act-unit", "identity activation layer (for testing purposes)", unit_activation_layer_t());
                nano::get_layers().add("act-tanh", "hyperbolic tangent activation layer", tanh_activation_layer_t());
                nano::get_layers().add("act-snorm", "x/sqrt(1+x^2) activation layer", snorm_activation_layer_t());
                nano::get_layers().add("act-splus", "soft-plus activation layer", softplus_activation_layer_t());
                nano::get_layers().add("affine", "fully-connected affine layer", affine_layer_t());
                nano::get_layers().add("conv", "convolution layer", convolution_layer_t());
        }

        static void init_models()
        {
                nano::get_models().add("forward-network", "feed-forward network", forward_network_t());
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

                        // register objects
                        init_tasks();
                        init_layers();
                        init_models();

                        init_losses();
                        init_criteria();
                        init_trainers();
                        init_batch_optimizers();
                        init_stoch_optimizers();
                }
        };

        static const init_t the_initializer;
}

