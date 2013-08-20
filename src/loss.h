#ifndef NANOCV_LOSS_H
#define NANOCV_LOSS_H

#include "core/manager.hpp"
#include "core/types.h"

namespace ncv
{
        // manage losses (register new ones, query and clone them)
        class loss_t;
        typedef manager_t<loss_t>               loss_manager_t;
        typedef loss_manager_t::robject_t       rloss_t;

        // classification convention
        inline scalar_t pos_target() { return +1.0; }
        inline scalar_t neg_target() { return -1.0; }
        vector_t class_target(size_t ilabel, size_t n_labels);

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
	
        class loss_t : public clonable_t<loss_t>
        {
        public:

                // destructor
                virtual ~loss_t() {}
                
                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const = 0;
                
                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const = 0;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const = 0;
        };
}

#endif // NANOCV_LOSS_H
