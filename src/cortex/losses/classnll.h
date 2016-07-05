#pragma once

#include "cortex/loss.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (single-class classification)
        /// NB: also called cross-entropy loss
        ///
        class classnll_loss_t : public loss_t
        {
        public:

                NANO_MAKE_CLONABLE(classnll_loss_t, "class negative log-likelihood loss")

                // constructor
                explicit classnll_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const override;
        };
}

