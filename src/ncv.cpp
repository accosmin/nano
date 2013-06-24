#include "ncv.h"
#include <cfenv>

#include "loss/loss_classnll.h"
#include "loss/loss_hinge.h"
#include "loss/loss_logistic.h"
#include "loss/loss_square.h"

#include "task/task_cifar10.h"
#include "task/task_mnist.h"
#include "task/task_stl10.h"
#include "task/task_cmufaces.h"

#include "model/model_conv_network.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void init()
        {
                // round to nearest integer
                std::fesetround(FE_TONEAREST);

                // register losses
                loss_manager_t::instance().add("classnll", classnll_loss_t());
                loss_manager_t::instance().add("hinge", hinge_loss_t());
                loss_manager_t::instance().add("logistic", logistic_loss_t());
                loss_manager_t::instance().add("square", square_loss_t());

                // register tasks
                task_manager_t::instance().add("mnist", mnist_task_t());
                task_manager_t::instance().add("cifar10", cifar10_task_t());
                task_manager_t::instance().add("stl10", stl10_task_t());
                task_manager_t::instance().add("cmu-faces", cmufaces_task_t());

                // register models
                model_manager_t::instance().add("convnet", conv_network_model_t());
        }

        //-------------------------------------------------------------------------------------------------
}
	
