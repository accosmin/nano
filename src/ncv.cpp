#include "ncv.h"

//#include "ncv_task_cifar10.h"
//#include "ncv_task_mnist.h"

#include "ncv_loss_classnll.h"
#include "ncv_loss_hinge.h"
#include "ncv_loss_logistic.h"
#include "ncv_loss_square.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void init()
        {
                // register tasks
//                task_manager::instance().add("mnist", mnist_task());
//                task_manager::instance().add("cifar10", cifar10_task());

                // register losses
                loss_manager::instance().add("classnll", classnll_loss());
                loss_manager::instance().add("hinge", hinge_loss());
                loss_manager::instance().add("logistic", logistic_loss());
                loss_manager::instance().add("square", square_loss());
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
	
