#pragma once

#include "loss.h"

namespace ncv
{
        ///
        /// \brief multi-class negative log-likelihood loss (single-label)
        ///
        class classnll_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(classnll_loss_t)

                // constructor
                classnll_loss_t(const string_t& parameters = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const;
        };
}

