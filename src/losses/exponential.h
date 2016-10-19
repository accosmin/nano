#pragma once

#include "classification_multi.h"

namespace nano
{
        ///
        /// \brief multi-class exponential loss: exp(sum(-targets_k * scores_k, k)).
        ///
        namespace detail
        {
                struct exponential_t
                {
                        static scalar_t value(const vector_t& targets, const vector_t& scores)
                        {
                                return std::exp((-targets.array() * scores.array()).sum());
                        }

                        static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                        {
                                return -targets.array() * std::exp((-targets.array() * scores.array()).sum());
                        }
                };
        }

        using exponential_loss_t = classification_multi_t<detail::exponential_t>;
}
