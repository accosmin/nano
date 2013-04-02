#ifndef NANOCV_H
#define NANOCV_H

#include "ncv_optimize.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include "ncv_timer.h"
#include "ncv_thread.h"
#include "ncv_image.h"
#include "ncv_stats.h"

namespace ncv
{
        // machine learning protocol
        enum class ml_protocol : int
        {
                train = 0,            // training
                valid,                // validation
                test                  // testing
        };

        namespace text
        {
                template <>
                inline string_t to_string(ml_protocol type)
                {
                        switch (type)
                        {
                        case ml_protocol::train:        return "train";
                        case ml_protocol::valid:        return "valid";
                        case ml_protocol::test:         return "test";
                        default:                        return "train";
                        }
                }

                template <>
                inline ml_protocol from_string<ml_protocol>(const string_t& string)
                {
                        if (string == "train")  return ml_protocol::train;
                        if (string == "valid")  return ml_protocol::valid;
                        if (string == "test")   return ml_protocol::test;
                        throw std::invalid_argument("Invalid data type <" + string + ">!");
                        return ml_protocol::train;
                }
        }

        // initialize library (register objects, start worker pool ...)
        void init();

//        // Labeling convention for classification
//	inline scalar_t pos_target() { return 1.0; }
//        inline scalar_t neg_target() { return -1.0; }
//        vector_t class_target(index_t ilabel, size_t n_labels);
        
//        // (Multivariate) regression and classification error
//        scalar_t l1_error(const vector_t& targets, const vector_t& scores);
//        scalar_t eclass_error(const vector_t& targets, const vector_t& scores);
//        scalar_t mclass_error(const vector_t& targets, const vector_t& scores);
	
//        // Compute the loss value and error for the given dataset
//        // FIXME: to be removed and replace with model_t::test(data, file)!
//	class dataset_t;
//	class model_t;
//	class loss_t;
//	void evaluate(const dataset_t& data, const loss_t& loss, model_t& model,
//		      scalar_t& lvalue, scalar_t& lerror);
}

#endif // NANOCV_H
