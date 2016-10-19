#pragma once

#include "classification_single.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        namespace detail
        {
                struct classnll_t
                {
                        static scalar_t value(const vector_t& targets, const vector_t& scores)
                        {
                                return  std::log(scores.array().exp().sum()) -
                                        scalar_t(0.5) * ((1 + targets.array()) * scores.array()).sum();
                        }

                        static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                        {
                                return  scores.array().exp() / (scores.array().exp().sum()) -
                                        scalar_t(0.5) * (1 + targets.array());
                        }
                };
        }

        using classnll_loss_t = classification_single_t<detail::classnll_t>;
}

