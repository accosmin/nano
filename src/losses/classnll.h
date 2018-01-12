#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        struct classnll_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        const auto classes = static_cast<scalar_t>(scores.size());
                        return  classes * std::log(scores.array().exp().sum()) -
                                scalar_t(0.5) * ((1 + targets.array()) * scores.array()).sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        const auto classes = static_cast<scalar_t>(scores.size());
                        return  classes * scores.array().exp() / (scores.array().exp().sum()) -
                                scalar_t(0.5) * (1 + targets.array());
                }
        };

        using classnll_loss_t = sclassification_t<classnll_t>;
}
