#include "nanocv.h"

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

#include "layers/layer_activation_unit.h"
#include "layers/layer_activation_tanh.h"
#include "layers/layer_activation_snorm.h"
#include "layers/layer_activation_splus.h"
#include "layers/layer_convolution.h"
#include "layers/layer_linear.h"
#include "layers/layer_pool.h"

#include "models/forward_network.h"

#include "trainers/batch_trainer.h"
#include "trainers/minibatch_trainer.h"
#include "trainers/stochastic_trainer.h"

#include "criteria/avg_criterion.h"
#include "criteria/avg_l2_criterion.h"
#include "criteria/avg_var_criterion.h"

#include <cfenv>

namespace ncv
{
        string_t version()
        {
                return  text::to_string(NANOCV_MAJOR_VERSION) + "." +
                        text::to_string(NANOCV_MINOR_VERSION) + "." +
                        text::to_string(NANOCV_REVISION_VERSION);
        }

        void init()
        {
                // round to nearest integer
                std::fesetround(FE_TONEAREST);

                // use Eigen with multiple threads
                Eigen::initParallel();

                // register losses
                loss_manager_t::instance().add("square", square_loss_t());
                loss_manager_t::instance().add("cauchy", cauchy_loss_t());
                loss_manager_t::instance().add("logistic", logistic_loss_t());
                loss_manager_t::instance().add("classnll", classnll_loss_t());

                // register tasks
                task_manager_t::instance().add("mnist", mnist_task_t());
                task_manager_t::instance().add("cifar10", cifar10_task_t());
                task_manager_t::instance().add("cifar100", cifar100_task_t());
                task_manager_t::instance().add("stl10", stl10_task_t());
                task_manager_t::instance().add("svhn", svhn_task_t());
                task_manager_t::instance().add("norb", norb_task_t());

                // register layers
                layer_manager_t::instance().add("act-unit", unit_activation_layer_t());
                layer_manager_t::instance().add("act-tanh", tanh_activation_layer_t());
                layer_manager_t::instance().add("act-snorm", snorm_activation_layer_t());
                layer_manager_t::instance().add("act-splus", softplus_activation_layer_t());
                layer_manager_t::instance().add("linear", linear_layer_t());                
                layer_manager_t::instance().add("conv", conv_layer_t());
                layer_manager_t::instance().add("pool-max", pool_max_layer_t());
                layer_manager_t::instance().add("pool-min", pool_min_layer_t());
                layer_manager_t::instance().add("pool-avg", pool_avg_layer_t());

                // register models
                model_manager_t::instance().add("forward-network", forward_network_t());

                // register trainers
                trainer_manager_t::instance().add("batch", batch_trainer_t());
                trainer_manager_t::instance().add("minibatch", minibatch_trainer_t());
                trainer_manager_t::instance().add("stochastic", stochastic_trainer_t());
                
                // register criteria
                criterion_manager_t::instance().add("avg", avg_criterion_t());
                criterion_manager_t::instance().add("l2n-reg", avg_l2_criterion_t());
                criterion_manager_t::instance().add("var-reg", avg_var_criterion_t());
        }

        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror)
        {
                sampler_t sampler(task);
                sampler.setup(fold).setup(sampler_t::atype::annotated);

                accumulator_t accumulator(model, 0, "avg", criterion_t::type::value, 0.0);
                accumulator.update(task, sampler.get(), loss);

                lvalue = accumulator.value();
                lerror = accumulator.avg_error();

                return accumulator.count();
        }
}
	
