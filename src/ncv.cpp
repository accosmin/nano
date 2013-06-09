#include "ncv.h"
#include <cfenv>

#include "ncv_loss_classnll.h"
#include "ncv_loss_hinge.h"
#include "ncv_loss_logistic.h"
#include "ncv_loss_square.h"

#include "ncv_task_cifar10.h"
#include "ncv_task_mnist.h"
#include "ncv_task_stl10.h"
#include "ncv_task_cmufaces.h"

#include "ncv_model_affine.h"

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
                model_manager_t::instance().add("affine", affine_model_t());
        }

        //-------------------------------------------------------------------------------------------------
}
	
