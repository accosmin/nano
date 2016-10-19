#pragma once

#include "classification_multi.h"

namespace nano
{
        ///
        /// \brief multi-class logistic loss: log(1 + exp(sum(-targets_k * scores_k, k))).
        ///
        namespace detail
        {
                struct logistic_t
                {
                        static scalar_t value(const vector_t& targets, const vector_t& scores)
                        {
                                return std::log1p(std::exp((-targets.array() * scores.array()).sum()));
                        }

                        static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                        {
                                return  -targets.array() * std::exp((-targets.array() * scores.array()).sum()) /
                                        (1 + std::exp((-targets.array() * scores.array()).sum()));
                        }
                };
        }

        using logistic_loss_t = classification_multi_t<detail::logistic_t>;
}
