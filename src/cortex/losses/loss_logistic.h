#pragma once

#include "cortex/loss.h"

namespace cortex
{
        ///
        /// \brief softmax multi-class logistic loss: softmax(log(1 + exp(-targets_k * scores_k)), k)
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                ZOB_MAKE_CLONABLE(logistic_loss_t, "multi-class logistic loss")

                // constructor
                explicit logistic_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const override;
        };
}
