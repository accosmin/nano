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
#include "tasks/task_iris.h"
#include "tasks/task_wine.h"

#include "layers/affine.h"
#include "layers/activation.h"
#include "layers/convolution.h"

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
#include "stoch/svrg.h"
#include "stoch/asgd.h"
#include "stoch/rmsprop.h"

#include "batch/gd.h"
#include "batch/cgd.h"
#include "batch/lbfgs.h"

#include "samplers/sampler_none.h"

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

        sampler_manager_t& get_samplers()
        {
                static sampler_manager_t manager;
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

        template <typename tobject>
        static auto maker()
        {
                return [] (const string_t& config) { return std::make_unique<tobject>(config); };
        }

        static void init_batch_optimizers()
        {
                auto& f = nano::get_batch_optimizers();
                f.add("gd", "gradient descent", maker<batch_gd_t>());
                f.add("cgd", "nonlinear conjugate gradient descent (default)", maker<batch_cgd_prp_t>());
                f.add("cgd-n", "nonlinear conjugate gradient descent (N)", maker<batch_cgd_n_t>());
                f.add("cgd-hs", "nonlinear conjugate gradient descent (HS)", maker<batch_cgd_hs_t>());
                f.add("cgd-fr", "nonlinear conjugate gradient descent (FR)", maker<batch_cgd_fr_t>());
                f.add("cgd-prp", "nonlinear conjugate gradient descent (PRP+)", maker<batch_cgd_prp_t>());
                f.add("cgd-cd", "nonlinear conjugate gradient descent (CD)", maker<batch_cgd_cd_t>());
                f.add("cgd-ls", "nonlinear conjugate gradient descent (LS)", maker<batch_cgd_ls_t>());
                f.add("cgd-dy", "nonlinear conjugate gradient descent (DY)", maker<batch_cgd_dy_t>());
                f.add("cgd-dycd", "nonlinear conjugate gradient descent (DYCD)", maker<batch_cgd_dycd_t>());
                f.add("cgd-dyhs", "nonlinear conjugate gradient descent (DYHS)", maker<batch_cgd_dyhs_t>());
                f.add("lbfgs", "limited-memory BFGS", maker<batch_lbfgs_t>());
        }

        static void init_stoch_optimizers()
        {
                auto& f = nano::get_stoch_optimizers();
                f.add("sg", "stochastic gradient (descent)", maker<stoch_sg_t>());
                f.add("sgm", "stochastic gradient (descent) with momentum", maker<stoch_sgm_t>());
                f.add("ngd", "stochastic normalized gradient", maker<stoch_ngd_t>());
                f.add("svrg", "stochastic variance reduced gradient", maker<stoch_svrg_t>());
                f.add("asgd", "averaged stochastic gradient (descent)", maker<stoch_asgd_t>());
                f.add("ag", "Nesterov's accelerated gradient", maker<stoch_ag_t>());
                f.add("agfr", "Nesterov's accelerated gradient with function value restarts", maker<stoch_agfr_t>());
                f.add("aggr", "Nesterov's accelerated gradient with gradient restarts", maker<stoch_aggr_t>());
                f.add("adam", "Adam (see citation)", maker<stoch_adam_t>());
                f.add("adagrad", "AdaGrad (see citation)", maker<stoch_adagrad_t>());
                f.add("adadelta", "AdaDelta (see citation)", maker<stoch_adadelta_t>());
                f.add("rmsprop", "RMSProp (see citation)", maker<stoch_rmsprop_t>());
        }

        static void init_trainers()
        {
                auto& f = nano::get_trainers();
                f.add("batch", "batch trainer", maker<batch_trainer_t>());
                f.add("stoch", "stochastic trainer", maker<stochastic_trainer_t>());
        }

        static void init_criteria()
        {
                auto& f = nano::get_criteria();
                f.add("avg", "average loss", maker<average_criterion_t>());
                f.add("avg-l2n", "L2-norm regularized average loss", maker<average_l2n_criterion_t>());
                f.add("avg-var", "variance-regularized average loss", maker<average_var_criterion_t>());
                f.add("max", "softmax loss", maker<softmax_criterion_t>());
                f.add("max-l2n", "L2-norm regularized softmax loss", maker<softmax_l2n_criterion_t>());
                f.add("max-var", "variance-regularized softmax loss", maker<softmax_var_criterion_t>());
        }

        static void init_losses()
        {
                auto& f = nano::get_losses();
                f.add("square",      "multivariate regression:     l(y, t) = 1/2 * L2(y, t)", maker<square_loss_t>());
                f.add("cauchy",      "multivariate regression:     l(y, t) = log(1 + L2(y, t))", maker<cauchy_loss_t>());
                f.add("logistic",    "multi-class classification:  l(y, t) = log(1 + exp(-t.dot(y)))", maker<logistic_loss_t>());
                f.add("classnll",    "single-class classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y)", maker<classnll_loss_t>());
                f.add("exponential", "multi-class classification:  l(y, t) = exp(-t.dot(y))", maker<exponential_loss_t>());
        }

        static void init_tasks()
        {
                auto& f = nano::get_tasks();
                f.add("mnist", "MNIST (1x28x28 digit classification)", maker<mnist_task_t>());
                f.add("cifar10", "CIFAR-10 (3x32x32 object classification)", maker<cifar10_task_t>());
                f.add("cifar100", "CIFAR-100 (3x32x32 object classification)", maker<cifar100_task_t>());
                f.add("stl10", "STL-10 (3x96x96 semi-supervised object classification)", maker<stl10_task_t>());
                f.add("svhn", "SVHN (3x32x32 digit classification in the wild)", maker<svhn_task_t>());
                f.add("iris", "IRIS (iris flower classification)", maker<iris_task_t>());
                f.add("wine", "WINE (wine classification)", maker<wine_task_t>());
                f.add("synth-charset", "synthetic character classification", maker<charset_task_t>());
        }

        static void init_layers()
        {
                auto& f = nano::get_layers();
                f.add("act-unit",  "activation: a(x) = x", maker<activation_layer_unit_t>());
                f.add("act-sin",   "activation: a(x) = sin(x)", maker<activation_layer_sine_t>());
                f.add("act-tanh",  "activation: a(x) = tanh(x)", maker<activation_layer_tanh_t>());
                f.add("act-splus", "activation: a(x) = log(1 + e^x)", maker<activation_layer_splus_t>());
                f.add("act-snorm", "activation: a(x) = x / sqrt(1 + x^2)", maker<activation_layer_snorm_t>());
                f.add("act-sigm",  "activation: a(x) = exp(x) / (1 + exp(x))", maker<activation_layer_sigm_t>());
                f.add("act-pwave", "activation: a(x) = x / (1 + x^2)", maker<activation_layer_pwave_t>());
                f.add("affine",    "transform:  L(x) = A * x + b", maker<affine_layer_t>());
                f.add("conv",      "transform:  L(x) = conv3D(x, kernel) + b", maker<convolution_layer_t>());
        }

        static void init_models()
        {
                auto& f = nano::get_models();
                f.add("forward-network", "feed-forward network", maker<forward_network_t>());
        }

        static void init_samplers()
        {
                auto& f = nano::get_samplers();
                f.add("none", "use samples as they are", maker<sampler_none_t>());
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
                        init_samplers();

                        init_losses();
                        init_criteria();
                        init_trainers();
                        init_batch_optimizers();
                        init_stoch_optimizers();
                }
        };

        static const init_t the_initializer;
}

