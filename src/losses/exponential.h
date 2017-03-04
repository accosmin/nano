#pragma once

#include "classification_multi.h"

namespace nano
{
        ///
        /// \brief multi-class exponential loss: exp(sum(-targets_k * scores_k, k)).
        ///
        struct exponential_t
        {
                static scalar_t value(const vector_t& targets, const vector_t& scores)
                {
                        return std::exp(-targets.dot(scores));
                }

                static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                {
                        return -targets.array() * std::exp(-targets.dot(scores));
                }
        };

        using exponential_loss_t = classification_multi_t<exponential_t>;
}
