#pragma once

#include "loss.h"

namespace nano
{
        ///
        /// \brief multi-class logistic loss: log(1 + exp(sum(-targets_k * scores_k, k)))
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                NANO_MAKE_CLONABLE(logistic_loss_t)

                // constructor
                explicit logistic_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const final;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const final;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const final;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const final;
        };
}
