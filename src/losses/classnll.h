#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        struct classnll_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return  std::log(outputs.array().exp().sum()) -
                                scalar_t(0.5) * ((1 + targets.array()) * outputs.array()).sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return  outputs.array().exp() / (outputs.array().exp().sum()) -
                                scalar_t(0.5) * (1 + targets.array());
                }
        };

        using classnll_loss_t = sclassification_t<classnll_t>;
}
