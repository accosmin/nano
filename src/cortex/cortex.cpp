#include "cortex.h"

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
#include "trainers/minibatch.h"
#include "trainers/stochastic.h"

#include "criteria/average.h"
#include "criteria/average_l2reg.h"
#include "criteria/average_varreg.h"

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
                        nano::get_trainers().add("minibatch", minibatch_trainer_t());
                        nano::get_trainers().add("stochastic", stochastic_trainer_t());

                        // register criteria
                        nano::get_criteria().add("avg", avg_criterion_t());
                        nano::get_criteria().add("l2n-reg", avg_l2_criterion_t());
                        nano::get_criteria().add("var-reg", avg_var_criterion_t());
                }
        };

        static const init_t the_initializer;
}

