#pragma once

#include "classification_multi.h"
#include "classification_single.h"

namespace nano
{
        ///
        /// \brief multi-class logistic loss: log(1 + exp(sum(-targets_k * scores_k, k))).
        ///
        struct logistic_t
        {
                static scalar_t value(const vector_t& targets, const vector_t& scores)
                {
                        return std::log1p(std::exp(-targets.dot(scores)));
                }

                static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                {
                        return  -targets.array() * std::exp(-targets.dot(scores)) /
                                (1 + std::exp((-targets.dot(scores))));
                }
        };

        using mlogistic_loss_t = classification_multi_t<logistic_t>;
        using slogistic_loss_t = classification_single_t<logistic_t>;
}
