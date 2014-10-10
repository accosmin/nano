#pragma once

#include "loss.h"

namespace ncv
{
        ///
        /// \brief multi-class logistic loss (single & multi-class classification)
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(logistic_loss_t, "multi-class logistic loss")

                // constructor
                logistic_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const;
        };
}
