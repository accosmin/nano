#include "ncv.h"

//#include "ncv_filter_edge.h"
//#include "ncv_filter_gauss.h"
//#include "ncv_filter_custom.h"
//#include "ncv_filter_sharpen.h"
//#include "ncv_filter_emboss.h"
//#include "ncv_filter_median.h"
//#include "ncv_filter_scale.h"
//#include "ncv_filter_mshift.h"

//#include "ncv_dataset.h"
//#include "ncv_math.h"

//#include "ncv_registerer.h"

//#include "ncv_task_cifar10.h"
//#include "ncv_task_mnist.h"

////#include "ncv_loss_classnll.h"
////#include "ncv_loss_hinge.h"
////#include "ncv_loss_logistic.h"
////#include "ncv_loss_square.h"

////#include "ncv_model_linear.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void init()
        {
//                // Initialize image filters
//                filter_manager::instance().add("grad_dx", grad_dx_filter());
//                filter_manager::instance().add("grad_dy", grad_dy_filter());
//                filter_manager::instance().add("grad_ld", grad_ld_filter());
//                filter_manager::instance().add("grad_rd", grad_rd_filter());

//                filter_manager::instance().add("edgeo", edgeo_filter());
//                filter_manager::instance().add("edgem", edgem_filter());
//                filter_manager::instance().add("gauss", gauss_filter());

//                filter_manager::instance().add("sharp_strong", sharp_strong_filter());
//                filter_manager::instance().add("sharp_smooth", sharp_smooth_filter());

//                filter_manager::instance().add("emboss", emboss_filter());
//                filter_manager::instance().add("median", median_filter());

//                filter_manager::instance().add("custom", custom_conv_filter());

//                filter_manager::instance().add("scale", scale_filter());
//                filter_manager::instance().add("mshift", mshift_filter());
        }

        //-------------------------------------------------------------------------------------------------

//        vector_t class_target(index_t ilabel, size_t n_labels)
//        {
//                vector_t target(n_labels);
//                target.setConstant(neg_target());
//                target[ilabel] = pos_target();
//                return target;
//        }

//        //-------------------------------------------------------------------------------------------------

//        scalar_t l1_error(const vector_t& targets, const vector_t& scores)
//        {
//                return (targets - scores).array().abs().sum();
//        }

//        //-------------------------------------------------------------------------------------------------
        
//        scalar_t eclass_error(const vector_t& targets, const vector_t& scores)
//        {
//                return (targets.array() * scores.array() <= std::numeric_limits<scalar_t>::epsilon()).count();
//        }

//        //-------------------------------------------------------------------------------------------------

//        scalar_t mclass_error(const vector_t& targets, const vector_t& scores)
//        {
//                std::ptrdiff_t idx = 0;
//                scores.maxCoeff(&idx);
                
//                return targets(idx) > 0.5 ? 0.0 : 1.0;
//        }

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

//        //-------------------------------------------------------------------------------------------------

//        void register_objects()
//        {
//                ncv::register_task("mnist", mnist_task_t());
//                ncv::register_task("cifar10", cifar10_task_t());

//                ncv::register_loss("classnll", classnll_loss_t());
//                ncv::register_loss("hinge", hinge_loss_t());
//                ncv::register_loss("logistic", logistic_loss_t());
//                ncv::register_loss("square", square_loss_t());

//                ncv::register_model("linear", linear_model_t());
//        }

        //-------------------------------------------------------------------------------------------------
}
	
