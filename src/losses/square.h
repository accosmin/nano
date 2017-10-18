#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief square loss: (target - score)^2.
        ///
        struct square_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        return scalar_t(0.5) * (scores - targets).array().square().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        return scores - targets;
                }
        };

        using square_loss_t = regression_t<square_t>;
}
