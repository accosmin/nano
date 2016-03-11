#pragma once

#include "cortex/loss.h"

namespace nano
{
        ///
        /// \brief softmax multi-class logistic loss: softmax(exp(-targets_k * scores_k), k)
        ///
        class exponential_loss_t : public loss_t
        {
        public:

                NANO_MAKE_CLONABLE(exponential_loss_t, "multi-class exponential loss")

                // constructor
                explicit exponential_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const override;
        };
}
