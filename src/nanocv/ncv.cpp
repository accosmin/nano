#include "ncv.h"

#include "losses/loss_classnll.h"
#include "losses/loss_logistic.h"
#include "losses/loss_square.h"

#include "tasks/task_cifar10.h"
#include "tasks/task_mnist.h"
#include "tasks/task_stl10.h"
#include "tasks/task_cbclfaces.h"

#include "layers/layer_linear.h"
#include "layers/layer_activation_unit.h"
#include "layers/layer_activation_tanh.h"
#include "layers/layer_activation_snorm.h"
#include "layers/layer_convolution.h"
#include "layers/layer_softmax_pool.h"
#include "layers/layer_softmax_abs_pool.h"

#include "models/forward_network.h"

#include "trainers/batch_trainer.h"
#include "trainers/minibatch_trainer.h"
#include "trainers/stochastic_trainer.h"

#include <cfenv>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        void init()
        {
                // round to nearest integer
                std::fesetround(FE_TONEAREST);

                // use Eigen with multiple threads
                Eigen::initParallel();

                // register losses
                loss_manager_t::instance().add("classnll", classnll_loss_t());
                loss_manager_t::instance().add("logistic", logistic_loss_t());
                loss_manager_t::instance().add("square", square_loss_t());

                // register tasks
                task_manager_t::instance().add("mnist", mnist_task_t());
                task_manager_t::instance().add("cifar10", cifar10_task_t());
                task_manager_t::instance().add("stl10", stl10_task_t());
                task_manager_t::instance().add("cbcl-faces", cbclfaces_task_t());

                // register layers
                layer_manager_t::instance().add("linear", linear_layer_t());
                layer_manager_t::instance().add("unit", unit_activation_layer_t());
                layer_manager_t::instance().add("tanh", tanh_activation_layer_t());
                layer_manager_t::instance().add("snorm", snorm_activation_layer_t());
                layer_manager_t::instance().add("conv", conv_layer_t());
                layer_manager_t::instance().add("smax-pool", softmax_pool_layer_t());
                layer_manager_t::instance().add("smax-abs-pool", softmax_abs_pool_layer_t());

                // register models
                model_manager_t::instance().add("forward-network", forward_network_t());

                // register trainers
                trainer_manager_t::instance().add("batch", batch_trainer_t());
                trainer_manager_t::instance().add("minibatch", minibatch_trainer_t());
                trainer_manager_t::instance().add("stochastic", stochastic_trainer_t());
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror)
        {
                lvalue = lerror = 0.0;
                size_t count = 0;

                const samples_t& samples = task.samples(fold);
                for (size_t i = 0; i < samples.size(); i ++)
                {
                        const sample_t& sample = samples[i];
                        const image_t& image = task.image(sample.m_index);

                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                const vector_t output = model.value(image, sample.m_region);

                                lvalue += loss.value(target, output);
                                lerror += loss.error(target, output);
                                ++ count;
                        }
                }

                if (count > 0)
                {
                        lvalue /= count;
                        lerror /= count;
                }

                return count;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
