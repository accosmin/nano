#pragma once

#include "nanocv/loss.h"

namespace ncv
{
        ///
        /// \brief multi-class negative log-likelihood loss (single-class classification)
        ///
        class classnll_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(classnll_loss_t, "multi-class negative log-likelihood loss")

                // constructor
                classnll_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const override;
        };
}

