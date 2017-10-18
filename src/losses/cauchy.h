#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief robust-to-noise Cauchy loss: log(1 + (target - score)^2).
        ///
        struct cauchy_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        return ((targets - scores).array().square() + 1).log().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores)
                {
                        return 2 * (scores - targets).array() / (1 + (scores - targets).array().square());
                }
        };

        using cauchy_loss_t = regression_t<cauchy_t>;
}
