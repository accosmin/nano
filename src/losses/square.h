#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief square loss: (target - score)^2.
        ///
        struct square_t
        {
                static scalar_t value(const vector_t& targets, const vector_t& scores)
                {
                        return scalar_t(0.5) * (scores - targets).array().square().sum();
                }

                static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                {
                        return scores - targets;
                }
        };

        using square_loss_t = regression_t<square_t>;
}
