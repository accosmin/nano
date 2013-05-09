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

#include "ncv_model_linear.h"

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
                task_manager_t::instance().add("cmufaces", cmufaces_task_t());

                // register models
                model_manager_t::instance().add("linear", linear_model_t());
        }

//        //-------------------------------------------------------------------------------------------------
        
//        void evaluate(const dataset_t& data, const loss_t& loss, model_t& model,
//		scalar_t& lvalue, scalar_t& lerror)
//	{
//		lvalue = lerror = 0.0;
//		for (size_t s = 0; s < data.n_samples(); s ++)
//		{
//			const vector_t& targets = data.targets(s);
//			const vector_t& scores = model.process(data.inputs(s));
			
//			lvalue += loss.value(targets, scores);
//			lerror += loss.error(targets, scores);
//		}
		
//                lvalue *= inversedata.n_samples());
//                lerror *= inversedata.n_samples());
//	}

        //-------------------------------------------------------------------------------------------------
}
	
