#ifndef NANOCV_LOSS_H
#define NANOCV_LOSS_H

#include "ncv_manager.h"

namespace ncv
{
        // manage losses (register new ones, query and clone them)
        class loss;
        typedef manager<loss>                   loss_manager;
        typedef loss_manager::robject_t         rloss;

        // classification convention
        inline scalar_t pos_target() { return +1.0; }
        inline scalar_t neg_target() { return -1.0; }
        vector_t class_target(index_t ilabel, size_t n_labels);

        // (multivariate) regression and classification error
        scalar_t l1_error(const vector_t& targets, const vector_t& scores);
        scalar_t eclass_error(const vector_t& targets, const vector_t& scores);
        scalar_t mclass_error(const vector_t& targets, const vector_t& scores);

        ////////////////////////////////////////////////////////////////////////////////
        // generic multivariate loss function of two parameters:
        //	the target value to predict and the current score estimation.
        // the loss function upper-bounds/approximates
        //      the true (usually non-smooth) error function to minimize.
        ////////////////////////////////////////////////////////////////////////////////
	
        class loss : public clonable<loss>
        {
        public:
                
                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const = 0;
                
                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const = 0;
                virtual scalar_t vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const = 0;
        };
}

#endif // NANOCV_LOSS_H
