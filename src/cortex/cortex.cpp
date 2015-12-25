#include "cortex.h"

#include "losses/loss_square.h"
#include "losses/loss_cauchy.h"
#include "losses/loss_logistic.h"
#include "losses/loss_classnll.h"

#include "tasks/task_mnist.h"
#include "tasks/task_cifar10.h"
#include "tasks/task_cifar100.h"
#include "tasks/task_stl10.h"
#include "tasks/task_svhn.h"
#include "tasks/task_norb.h"
#include "tasks/task_charset.h"
#include "tasks/task_random.h"

#include "layers/layer_activation_unit.h"
#include "layers/layer_activation_tanh.h"
#include "layers/layer_activation_snorm.h"
#include "layers/layer_activation_splus.h"
#include "layers/layer_convolution.h"
#include "layers/layer_affine.h"
#include "layers/layer_pool.h"
#include "layers/layer_plane_affine.h"

#include "models/forward_network.h"

#include "trainers/batch_trainer.h"
#include "trainers/minibatch_trainer.h"
#include "trainers/stochastic_trainer.h"

#include "criteria/avg_criterion.h"
#include "criteria/avg_l2_criterion.h"
#include "criteria/avg_var_criterion.h"

#include <cfenv>

namespace cortex
{
        void init()
        {
                // round to nearest integer
                std::fesetround(FE_TONEAREST);

                // use Eigen with multiple threads
                Eigen::initParallel();
                Eigen::setNbThreads(0);

                // register losses
                cortex::get_losses().add("square", square_loss_t());
                cortex::get_losses().add("cauchy", cauchy_loss_t());
                cortex::get_losses().add("logistic", logistic_loss_t());
                cortex::get_losses().add("classnll", classnll_loss_t());

                // register tasks
                cortex::get_tasks().add("mnist", mnist_task_t());
                cortex::get_tasks().add("cifar10", cifar10_task_t());
                cortex::get_tasks().add("cifar100", cifar100_task_t());
                cortex::get_tasks().add("stl10", stl10_task_t());
                cortex::get_tasks().add("svhn", svhn_task_t());
                cortex::get_tasks().add("norb", norb_task_t());
                cortex::get_tasks().add("charset", charset_task_t());
                cortex::get_tasks().add("random", random_task_t());

                // register layers
                cortex::get_layers().add("act-unit", unit_activation_layer_t());
                cortex::get_layers().add("act-tanh", tanh_activation_layer_t());
                cortex::get_layers().add("act-snorm", snorm_activation_layer_t());
                cortex::get_layers().add("act-splus", softplus_activation_layer_t());
                cortex::get_layers().add("affine", affine_layer_t());
                cortex::get_layers().add("plane-affine", plane_affine_layer_t());
                cortex::get_layers().add("conv", conv_layer_t());
                cortex::get_layers().add("pool-max", pool_max_layer_t());
                cortex::get_layers().add("pool-min", pool_min_layer_t());
                cortex::get_layers().add("pool-avg", pool_avg_layer_t());

                // register models
                cortex::get_models().add("forward-network", forward_network_t());

                // register trainers
                cortex::get_trainers().add("batch", batch_trainer_t());
                cortex::get_trainers().add("minibatch", minibatch_trainer_t());
                cortex::get_trainers().add("stochastic", stochastic_trainer_t());
                
                // register criteria
                cortex::get_criteria().add("avg", avg_criterion_t());
                cortex::get_criteria().add("l2n-reg", avg_l2_criterion_t());
                cortex::get_criteria().add("var-reg", avg_var_criterion_t());
        }
}
	
