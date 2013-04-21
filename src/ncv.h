#ifndef NANOCV_H
#define NANOCV_H

#include "ncv_optimize.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include "ncv_timer.h"
#include "ncv_thread.h"
#include "ncv_image.h"
#include "ncv_stats.h"
#include "ncv_loss.h"

namespace ncv
{
        // initialize library (register objects, start worker pool ...)
        void init();


	
//        // Compute the loss value and error for the given dataset
//        // FIXME: to be removed and replace with model_t::test(data, file)!
//	class dataset_t;
//	class model_t;
//	class loss_t;
//	void evaluate(const dataset_t& data, const loss_t& loss, model_t& model,
//		      scalar_t& lvalue, scalar_t& lerror);
}

#endif // NANOCV_H
