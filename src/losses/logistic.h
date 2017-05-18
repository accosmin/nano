#pragma once

#include "classification_multi.h"
#include "classification_single.h"

namespace nano
{
        ///
        /// \brief multi-class logistic loss: sum(log(1 + exp(-targets_k * scores_k)), k).
        ///
        struct logistic_t
        {
                static scalar_t value(const vector_t& targets, const vector_t& scores)
                {
                        return  (1 + (-targets.array() * scores.array()).exp()).log().sum();
                }

                static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                {
                        return  -targets.array() * (-targets.array() * scores.array()).exp() /
                                (1 + (-targets.array() * scores.array()).exp());
                }
        };

        using mlogistic_loss_t = classification_multi_t<logistic_t>;
        using slogistic_loss_t = classification_single_t<logistic_t>;
}
